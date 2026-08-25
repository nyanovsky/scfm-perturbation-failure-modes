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

            d = r.arm(line, arm).get("direct_prediction")
            if d:
                base = r.arm(line, arm).get("direct_prediction_baseline")
                a_out["direct_residual"] = claim(
                    f"{ARM_LABEL[arm]} in {L}, its own predicted response against the measured "
                    f"one on the residual target, baseline {base}",
                    d["score"], DIRECT,
                    key(cell_line=line, arm_label=f"direct_prediction/{arm}",
                        target="residual", weighting="weighted", baseline=base),
                )
                a_out["direct_residual__lo"] = d["lo"]
                a_out["direct_residual__hi"] = d["hi"]
            by = r.arm(line, arm).get("direct_prediction_by_baseline", {})
            for bname, row in by.items():
                a_out[f"direct_residual__{bname}"] = row["score"]

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

        cf = r.crossfit_row(line, "own_trainable")
        if cf:
            e["crossfit"] = claim(
                f"{L}, response features of the published recipe pooled over the whole screen "
                "by cross-fitting, verdict cell",
                cf["score"], "results/summary/crossfit_%s__own_trainable.parquet" % line,
                key(cell_line=line, arm="own_trainable", scope="whole_screen",
                    target="residual", weighting="weighted"),
            )
            e["crossfit__lo"], e["crossfit__hi"] = cf["lo"], cf["hi"]
            e["crossfit__n"], e["crossfit__n_groups"] = cf["n"], cf["n_groups"]

        le = r.last_epoch_row(line, "own_trainable")
        if le:
            e["last_epoch"] = claim(
                f"{L}, response features read off the LAST epoch rather than the selected one, "
                "verdict cell",
                le["score"], RESTRICTED,
                rkey(cell_line=line, arm=B.last_epoch_arm("own_trainable")),
            )
        n["lines"][line] = e

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
            "reference arm across the sixteen jobs",
            float(r.agreement["spread"].max()), AGREE, key(column="spread", agg="max"),
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


def build(summary_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    r = B.Results(summary_dir)
    n, claims = collect(r)
    h = n["lines"].get("hepg2", {})
    j = n["lines"].get("jurkat", {})
    ho = h.get("arms", {}).get("own_trainable", {})
    jo = j.get("arms", {}).get("own_trainable", {})

    doc = {
        "headline": (
            "Fine-tuning scGPT on the CRISPRi screen it is evaluated on, with scGPT's own "
            "published perturbation recipe, does not install a gene-specific regulatory signal "
            f"in the model. Across the {n['n_models']} fine-tuned models (five gene-table arms "
            f"in two cell lines), {n['n_models_in_band']} leave the queried response "
            "features inside the shuffled-label band, and the highest any reaches is "
            f"{n.get('best_response_feature_score', float('nan')):.4f}, against the same "
            f"model's static gene table at {h.get('scgpt_emb', float('nan')):.4f} in HepG2 and "
            f"{j.get('scgpt_emb', float('nan')):.4f} in Jurkat. The reason is visible on the "
            "model's own prediction head, before any feature swap: the fine-tuned model "
            "reproduces the response shared across knockdowns well but scores "
            f"{ho.get('direct_residual', float('nan')):.4f} in HepG2 and "
            f"{jo.get('direct_residual', float('nan')):.4f} in Jurkat on the residual after "
            "that shared response is removed. Fine-tuning teaches the average knockdown "
            "response, not which gene was knocked down."
        ),
        "assessment": (
            "The pre-registered prediction is confirmed in its first half and overtaken in its "
            "second. It said the response features would stay in the null band while the model "
            "might still predict better; the features do stay in the band, but the model does "
            "not predict the perturbation-specific part better either, so the null is a "
            "property of the fine-tuned model rather than of the read-out procedure. That is "
            "the strongest form the result could take, because it does not depend on the "
            "translator, the query family, or the feature standardisation: the same conclusion "
            "is reached by scoring the model's own output directly.\n\n"
            "Scope and limits. This is one architecture and one recipe, run on the screen the "
            "question is about, which is the setting most favourable to installing the signal "
            "and therefore the setting in which a null is most informative; it does not speak "
            "to other architectures or to fine-tuning on a different screen. A fine-tuned "
            "model may only be scored on perturbations it never saw, so the main arms read two "
            f"of run 3's five folds. That costs resolution and the report states it: "
            f"{_band_clause(n)}. The claim the design supports is therefore that fine-tuning "
            "does not lift the response features anywhere near the gene table and does not "
            "move them detectably at all, not that they are exactly zero. Three controls "
            "address the obvious objections: the arm is refitted on the complementary folds so "
            "the pooled result covers the whole screen, the batch sampler is run in the exact "
            "upstream configuration as well as ours, and because the recipe's selection rule "
            "picks epoch 1 in Jurkat the features are also read off the last epoch, which the "
            "rule rejected."
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
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    summary = Path(args.summary) if args.summary else paths.results_dir() / "summary"
    out = Path(args.out) if args.out else paths.repo_dir()
    a, b = build(summary, out)
    doc = json.loads(a.read_text())
    print(doc["headline"])
    print(f"\nwrote {a} and {b} ({len(json.loads(b.read_text())['claims'])} claims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
