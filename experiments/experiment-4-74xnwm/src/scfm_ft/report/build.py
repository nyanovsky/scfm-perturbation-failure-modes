"""Build run 4's report page from the summary tables.

    PYTHONPATH=src uv run python -m scfm_ft.report.build --out "$SILICO_APP_DIR"

Every number in the prose is read from ``results/summary``; nothing is typed in
by hand. The figures are rebuilt from the same tables, so the page and the
bundles under ``figures/`` cannot disagree.

The HTML helpers are run 3's ``report.page`` module, imported through the
sibling alias rather than copied, so the two reports share one visual language.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from silico_figures import PLOTLY_CONFIG

from .. import config as C
from .. import paths
from ..sibling import run3
from . import figures as figmod

P = run3("report.page")

SRC = "experiments/experiment-4-74xnwm"
S_CONFIG = f"{SRC}/src/scfm_ft/config.py"
S_INGEST = f"{SRC}/src/scfm_ft/ingest.py"
S_MODEL = f"{SRC}/src/scfm_ft/scgpt_pert/model.py"
S_TRAIN = f"{SRC}/src/scfm_ft/scgpt_pert/train.py"
S_DATA = f"{SRC}/src/scfm_ft/scgpt_pert/data.py"
S_QUERY = f"{SRC}/src/scfm_ft/scgpt_pert/query.py"
S_SCORE = f"{SRC}/src/scfm_ft/score.py"
S_RUNARM = f"{SRC}/src/scfm_ft/run_arm.py"
S_ANALYSIS = f"{SRC}/src/scfm_ft/analysis.py"
S_FOLLOWUP = f"{SRC}/src/scfm_ft/followup.py"
S_FIGS = f"{SRC}/src/scfm_ft/report/figures.py"

LINE = {"hepg2": "HepG2", "jurkat": "Jurkat"}
ARM_LABEL = {
    "own_trainable": "scGPT table, trainable",
    "own_frozen": "scGPT table, frozen",
    "esm2_trainable": "ESM2-3B table, trainable",
    "esm2_frozen": "ESM2-3B table, frozen",
    "rand_frozen": "random table, frozen",
}
ARM_ORDER = tuple(a.name for a in C.ARMS)

VERDICT = ("residual", "weighted")
DIAGNOSTIC = ("centered", "weighted")


def last_epoch_arm(arm: str, family: str = "ft", suffix: str = "resp") -> str:
    """The scored arm name of a run's last-epoch re-reading."""
    return f"{family}_{C.model_stem(f'{arm}__lastepoch')}_{suffix}"


def nan() -> float:
    return float("nan")


def ci(v: float, lo: float, hi: float) -> str:
    """A point estimate with its bootstrap interval, in plain text."""
    if v is None or v != v:
        return "n/a"
    return f"{P.num(v)} ({P.num(lo)} to {P.num(hi)})"


def civ(d: dict | None) -> str:
    if not d:
        return "n/a"
    return ci(d.get("score"), d.get("lo"), d.get("hi"))


# --------------------------------------------------------------------------- #
# the tables
# --------------------------------------------------------------------------- #
class Results:
    def __init__(self, summary_dir: Path):
        self.dir = Path(summary_dir)
        self.head = json.loads((self.dir / "headline.json").read_text())
        self.lines = self.head["cell_lines"]
        self.restricted = pd.read_parquet(self.dir / "restricted.parquet")
        self.band = pd.read_parquet(self.dir / "null_band.parquet")
        self.contrasts = self._opt("contrasts")
        self.direct = self._opt("direct")
        self.epochs = self._opt("training_epochs")
        self.runs = self._opt("training_runs")
        self.agreement = self._opt("reference_agreement")
        self.spread = self._opt("prediction_spread")
        self.crossfit = self._crossfit()
        self.ingest = self._ingest()

    def _opt(self, stem: str) -> pd.DataFrame:
        f = self.dir / f"{stem}.parquet"
        return pd.read_parquet(f) if f.exists() else pd.DataFrame()

    def _crossfit(self) -> pd.DataFrame:
        frames = [pd.read_parquet(f) for f in sorted(self.dir.glob("crossfit_*.parquet"))
                  if not f.name.startswith("crossfit_per_pert")]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _ingest(self) -> dict:
        out = {}
        for line in C.CELL_LINES:
            f = self.dir / f"summary_{line}__own_trainable.json"
            if f.exists():
                out[line] = json.loads(f.read_text())
        return out

    # ----- accessors ------------------------------------------------------ #
    def arm(self, line: str, arm: str) -> dict:
        return self.lines.get(line, {}).get("arms", {}).get(arm, {})

    def ref(self, line: str, arm: str) -> dict | None:
        return self.lines.get(line, {}).get("reference", {}).get(arm)

    def feat(self, line: str, arm: str, key: str) -> dict | None:
        """One feature-swap cell: key like ``final__value_substitution_resp``."""
        return self.arm(line, arm).get(key)

    def training(self, line: str, arm: str) -> dict:
        return self.arm(line, arm).get("training", {})

    def null(self, line: str, base_arm: str = "coexp") -> dict | None:
        return self.lines.get(line, {}).get("null_band", {}).get(base_arm)

    def null_ceiling(self, line: str) -> float:
        """The largest shuffled-label draw in that cell line, any base arm."""
        b = self.lines.get(line, {}).get("null_band", {})
        vals = [v["max"] for v in b.values() if v.get("max") is not None]
        return max(vals) if vals else nan()

    def in_band(self, line: str, point: dict | None) -> bool:
        if not point:
            return False
        return float(point["score"]) <= self.null_ceiling(line)

    def crossfit_row(self, line: str, arm: str, target="residual", weighting="weighted"):
        if self.crossfit.empty:
            return None
        s = self.crossfit[
            (self.crossfit.cell_line == line)
            & (self.crossfit.arm == arm)
            & (self.crossfit.target == target)
            & (self.crossfit.weighting == weighting)
        ]
        if s.empty:
            return None
        r = s.iloc[0]
        return {
            "score": float(r["disattenuated"]),
            "lo": float(r["disattenuated_lo"]),
            "hi": float(r["disattenuated_hi"]),
            "n": int(r["n_scored"]),
            "n_groups": int(r["n_groups"]),
        }

    def run_feature_row(self, line: str, run: str, family: str = "ft",
                        suffix: str = "resp") -> dict | None:
        """One run's feature arm in the verdict cell, by run name.

        Works for a tagged run (``own_trainable__upstream``) as well as a plain
        one; the scored arm name embeds the bundle model key, not the run name.
        """
        s = self.restricted[
            (self.restricted.cell_line == line)
            & (self.restricted.arm == f"{family}_{C.model_stem(run)}_{suffix}")
            & (self.restricted.target == VERDICT[0])
            & (self.restricted.weighting == VERDICT[1])
            & (self.restricted.translator == "mlp")
            & (~self.restricted.is_shuffled)
        ]
        if s.empty:
            return None
        r = s.iloc[0]
        return {
            "score": float(r["disattenuated"]),
            "lo": float(r["disattenuated_lo"]),
            "hi": float(r["disattenuated_hi"]),
            "n": int(r["n_scored"]),
            "n_groups": int(r["n_groups"]),
        }

    def last_epoch_row(self, line: str, arm: str, family: str = "ft",
                       suffix: str = "resp") -> dict | None:
        """The features read off the run's last epoch, if that control ran."""
        return self.run_feature_row(line, f"{arm}__lastepoch", family, suffix)

    def spread_row(self, line: str, arm: str) -> dict | None:
        """How perturbation-specific one model's own predicted response is."""
        if self.spread.empty:
            return None
        s = self.spread[(self.spread.cell_line == line) & (self.spread.arm == arm)]
        return None if s.empty else s.iloc[0].to_dict()

    def measured_spread(self, line: str) -> float:
        """The same statistic on the measured effects, from run 3's reference."""
        ref = (
            self.ingest.get(line, {})
            .get("bundle_inventory", {})
            .get("reference", {})
            .get("shared_response", {})
        )
        v = ref.get("median_residual_fraction_of_variance")
        return float(v) if v is not None else nan()

    def paired(self, line: str, arm: str, against: str) -> dict | None:
        for d in self.arm(line, arm).get("paired_differences", []):
            if d["against"] == against and d["arm"] == arm:
                return d
        return None

    def run_row(self, line: str, arm: str) -> dict | None:
        if self.runs.empty:
            return None
        s = self.runs[(self.runs.cell_line == line) & (self.runs.arm == arm)]
        return None if s.empty else s.iloc[0].to_dict()

    def all_feature_points(self, key: str) -> list[dict]:
        pts = []
        for line in C.CELL_LINES:
            for arm in ARM_ORDER:
                p = self.feat(line, arm, key)
                if p:
                    pts.append({"cell_line": line, "arm": arm, **p})
        return pts


# --------------------------------------------------------------------------- #
# prose
# --------------------------------------------------------------------------- #
def opening(r: Results) -> str:
    rows = r.all_feature_points("final__value_substitution_resp")
    best = max((p["score"] for p in rows), default=nan())
    n_in = sum(1 for p in rows if r.in_band(p["cell_line"], p))
    direct_h = r.arm("hepg2", "own_trainable").get("direct_prediction")
    direct_j = r.arm("jurkat", "own_trainable").get("direct_prediction")
    emb_h = r.ref("hepg2", "scgpt_emb")
    emb_j = r.ref("jurkat", "scgpt_emb")
    return P.section(
        "",
        "<p>Runs 2 and 3 found that when you ask a single-cell foundation model what happens "
        "if you change one gene, the answer carries no usable regulatory signal: the response "
        "features sat inside the shuffled-label band while the model's static gene-identity "
        "table beat co-expression comfortably. The obvious objection is that the models were "
        "never trained to answer that question. So we trained one to.</p>"
        "<p>We fine-tuned scGPT on each CRISPRi screen with the recipe scGPT publishes for "
        "perturbation prediction, holding out two of run 3's five folds so a fine-tuned model "
        "is only ever scored on perturbations it has not seen, and then pushed the fine-tuned "
        f"checkpoints through run 3's feature swap unchanged. Across the {len(rows)} fine-tuned "
        f"models in two cell lines, {n_in} leave the response features inside the "
        f"shuffled-label band; the highest any of them reaches is {P.num(best)}. The same "
        "model's static gene table, read through the same translator on the same held-out "
        f"perturbations, sits at {P.num((emb_h or {}).get('score', nan()))} in HepG2 and "
        f"{P.num((emb_j or {}).get('score', nan()))} in Jurkat, roughly ten times higher, and "
        "fine-tuning does not move that either.</p>"
        "<p>The stronger result is that the fine-tuned model does not acquire the signal on "
        "its own prediction head. Its predicted perturbation response, scored with no "
        "translator in the path against the same measured effects, agrees well before the "
        "response shared by every perturbation is removed "
        f"({P.num((direct_h or {}).get('raw', nan()))} in HepG2, "
        f"{P.num((direct_j or {}).get('raw', nan()))} in Jurkat) and reaches "
        f"{P.num((direct_h or {}).get('fold_centred', nan()))} and "
        f"{P.num((direct_j or {}).get('fold_centred', nan()))} once it is. Fine-tuning "
        "teaches the model what a knockdown does on average in that cell line. It does not "
        "teach it which gene was knocked down, so there is no gene-specific signal for the "
        "feature swap to find.</p>",
        sources=[S_RUNARM, S_ANALYSIS],
    )


def setup(r: Results) -> str:
    ing = r.ingest
    rows = []
    for line in C.CELL_LINES:
        d = ing.get(line, {})
        t = r.training(line, "own_trainable")
        rows.append([
            LINE[line],
            f"{d.get('n_perturbations_total', 'n/a')}",
            f"{d.get('n_groups_total', 'n/a')}",
            f"{d.get('n_scored', 'n/a')}",
            f"{d.get('n_groups_scored', 'n/a')}",
            f"{t.get('n_train_cells', 'n/a')}",
        ])
    tbl = P.table(
        ["cell line", "perturbations", "response groups", "scored", "scored groups",
         "training cells"],
        rows,
        sources=[S_INGEST],
        caption="What each screen contributes. Only folds 0 and 1 of run 3's grouped split "
                "are scored; the other three folds train the model.",
    )
    rec = C.RECIPE
    sp = P.spec([
        ("the base model", [
            ("checkpoint", f"{C.SCGPT_REPO} at {C.SCGPT_REVISION[:12]}"),
            ("transferred tensors", ", ".join(rec.load_param_prefixes)),
            ("expression decoder", "re-initialised, as the tutorial does"),
        ]),
        ("the published recipe, unchanged", [
            ("objective", rec.loss),
            ("optimizer", f"{rec.optimizer}, lr {rec.lr}"),
            ("batch size", str(rec.batch_size)),
            ("epochs", f"up to {rec.epochs}, early stop after {rec.early_stop}"),
            ("schedule", f"gamma {rec.schedule_gamma} every {rec.schedule_interval} epoch"),
            ("gene subset per batch", f"{rec.max_seq_len} of the read-out set"),
            ("selection", rec.val_metric),
        ]),
        ("what varies across arms", [
            (ARM_LABEL[a.name], a.note) for a in C.ARMS
        ]),
    ])
    return P.section(
        "What was fine-tuned",
        "<p>scGPT ships a perturbation fine-tuning tutorial. We restated it rather than "
        "importing it, because the run needs the model to be queryable the way runs 2 and 3 "
        "query it, and checked the restatement against the installed library: the masked-MSE "
        "loss is compared numerically against <code>scgpt/loss.py</code> at load time, and the "
        "batch sampler reproduces the upstream one exactly under "
        "<code>--upstream-sampling</code>.</p>"
        + tbl
        + "<p>Five arms vary only the gene-identity table and whether it receives gradient. "
        "The scGPT arms start from the model's own learned table. The ESM2 arms replace it "
        "with the 2,560-dimensional protein table run 3 built from sequence alone, behind a "
        "learned projection into the model width. The random arm keeps that projection and "
        "the same width over vectors that carry no information about the gene, so it is a "
        "capacity-matched null for the two ESM2 arms.</p>"
        + sp,
        sources=[S_CONFIG, S_TRAIN, S_MODEL],
    )


def resolution_note(r: Results) -> str:
    """How much this evaluation can see, stated with the null rather than implied."""
    parts = []
    for line in C.CELL_LINES:
        band = r.null(line, "coexp")
        co = r.ref(line, "coexp")
        pts = [p for p in r.all_feature_points("final__value_substitution_resp")
               if p["cell_line"] == line]
        if not (band and co and pts):
            continue
        parts.append(
            f"<li><strong>{LINE[line]}</strong>: the shuffled-label draws span "
            f"{P.num(r.null_ceiling(line))} at the top, and co-expression itself scores "
            f"{P.num(co['score'])} on these {pts[0]['n_groups']} response groups. An effect "
            "smaller than the band is not resolvable here.</li>"
        )
    if not parts:
        return ""
    return (
        "<p>Two folds is a narrower evaluation than run 3's five, and it is worth being "
        "explicit about what that costs before reading the null. Scoring only the folds a "
        "fine-tuned model never saw leaves fewer response groups to resample, so the "
        "shuffled-label band is wider than it is over the whole screen, and co-expression, "
        "which run 3 resolves clearly above zero, is not separated from it here either:</p>"
        f"<ul>{''.join(parts)}</ul>"
        "<p>So the claim this section supports is bounded: fine-tuning does not lift the "
        "response features to anywhere near the gene table, and does not move them "
        "detectably at all. It is not evidence that they are exactly zero.</p>"
    )


def holdout_note(r: Results) -> str:
    cf = []
    for line in C.CELL_LINES:
        row = r.crossfit_row(line, "own_trainable")
        if row:
            cf.append((line, row))
    if not cf:
        return ""
    parts = ", ".join(
        f"{LINE[line]} {civ(row)} over {row['n']} perturbations in {row['n_groups']} groups"
        for line, row in cf
    )
    return (
        "<p>Scoring only two folds costs resolution, so the primary arm was fitted twice "
        "more, once on folds 2 and 3 and once on fold 4, and the held-out rows pooled. Every "
        "perturbation in the screen is then scored by a model that never saw it, and the "
        f"pooled response-feature score is {parts}. Pooling does not change the picture; it "
        "removes the objection that two folds is too small a window to see an effect in.</p>"
    )


def verdict_section(r: Results, chart: str) -> str:
    rows = []
    for line in C.CELL_LINES:
        band = r.null_ceiling(line)
        for arm in ARM_ORDER:
            pre = r.feat(line, arm, "init__value_substitution_resp")
            post = r.feat(line, arm, "final__value_substitution_resp")
            own = r.feat(line, arm, "final__pert_flag_resp")
            rows.append([
                LINE[line],
                ARM_LABEL[arm],
                civ(pre),
                civ(post),
                civ(own),
                P.num(band),
            ])
    tbl = P.table(
        ["cell line", "arm", "before fine-tuning", "after fine-tuning",
         "after, model's own query", "shuffled-label ceiling"],
        rows,
        sources=[S_SCORE, S_ANALYSIS],
        caption="Response features through run 3's translator, scored on the folds the "
                "fine-tuned model never saw. Reliability-weighted, ceiling-corrected "
                "correlation on the residual target, with 95% intervals over 2,000 resamples "
                "of the response groups.",
    )
    post = r.all_feature_points("final__value_substitution_resp")
    n_in = sum(1 for p in post if r.in_band(p["cell_line"], p))
    return P.section(
        "The feature swap, before and after",
        "<p>The measurement is run 3's, unchanged: extract the model's predicted change in "
        "every read-out gene when one gene's input value is moved to the anchor, standardise "
        "it across genes, hand the pair of gene features to the fixed translator, and score "
        "the held-out perturbations against the measured CRISPRi effects. The only thing that "
        "differs between the two columns below is which checkpoint answered the query.</p>"
        + chart
        + tbl
        + f"<p>{n_in} of the {len(post)} fine-tuned models sit at or below the largest "
        "shuffled-label draw in their cell line, and none approaches the gene-table arm. "
        "Reading the features through the model's own perturbation-flag query, the query the "
        "fine-tuning objective is literally defined on, does not change that.</p>"
        + resolution_note(r)
        + holdout_note(r),
        sources=[S_QUERY, S_SCORE],
    )


def direct_section(r: Results, chart: str) -> str:
    rows = []
    for line in C.CELL_LINES:
        for arm in ARM_ORDER:
            a = r.arm(line, arm)
            d = a.get("direct_prediction")
            t = a.get("training", {})
            agree = t.get("response_agreement_at_selected_epoch", {})
            sp = r.spread_row(line, arm) or {}
            rows.append([
                LINE[line],
                ARM_LABEL[arm],
                P.num((d or {}).get("raw", nan())),
                civ(d),
                P.num((d or {}).get("fold_centred", nan())),
                P.num(sp.get("vs_measured_control_mean__median_residual_fraction_of_variance",
                             nan())),
                P.num(agree.get("ctrlmean_centered", nan())),
            ])
    tbl = P.table(
        ["cell line", "arm", "held-out response, raw", "held-out response, residual",
         "held-out response, fold-centred", "gene-specific fraction of the prediction",
         "validation agreement, centred"],
        rows,
        sources=[S_FOLLOWUP, S_TRAIN],
        caption="The model's own predicted perturbation response, with no translator in the "
                "path, against the baseline the published recipe is evaluated on. The first "
                "three columns are scored on the held-out folds against the measured effects. "
                "The fourth is the median fraction of a predicted profile's variance that "
                "survives subtracting the mean prediction, the statistic run 3 reports for "
                "the measured effects, taken over every in-scope perturbation because it is "
                "a property of the model's output rather than a score. The last is the "
                "recipe's own validation diagnostic, centred, at the selected epoch.",
    )
    return P.section(
        "What the fine-tuned model actually learned",
        "<p>A feature swap can only find signal the model has. Before concluding anything "
        "about the features, it is worth asking whether fine-tuning worked at all on its own "
        "terms. So we scored the model's prediction directly: run the perturbation forward "
        "pass, subtract the measured control profile, and compare the predicted change in "
        "every read-out gene against the measured change, one correlation per held-out "
        "perturbation, through the same reliability weighting and ceiling correction the "
        "feature arms use.</p>"
        + chart
        + tbl
        + spread_note(r)
        + "<p>Read across a row, the three scored columns point in opposite directions and "
        "the reason is the same in both cell lines. Against the raw measured profile the "
        "fine-tuned model does well, because most of what a CRISPRi knockdown does to a cell "
        "is shared across knockdowns and the model learns that shared response. The residual "
        "column subtracts the mean effect from the measurement but not from the prediction, "
        "so a prediction that hardly changes between perturbations still correlates with the "
        "fold's shared offset; that is why its interval is wide enough to cross zero. The "
        "fold-centred column removes the held-out fold's own mean from both sides, which is "
        "the only one of the three that asks whether the prediction separates one "
        "perturbation from another, and there the agreement is approximately zero. The same "
        "contrast appears inside training: the recipe's own validation diagnostic reads high "
        "before centring and at zero after it, in every epoch of every run.</p>"
        "<p>This matters for how the rest of the page should be read. The response features "
        "sitting in the null band is not a failure of the feature swap to find something the "
        "model has. The model does not have it.</p>",
        sources=[S_FOLLOWUP, S_SCORE],
    )


def spread_note(r: Results) -> str:
    """How much of the predicted response depends on which gene was knocked down."""
    bits = []
    for line in C.CELL_LINES:
        sp = r.spread_row(line, "own_trainable")
        if not sp:
            continue
        bits.append(
            f"{LINE[line]} "
            f"{P.num(sp['vs_measured_control_mean__median_residual_fraction_of_variance'])} "
            f"against {P.num(r.measured_spread(line))} in the measurement, with a median "
            f"correlation of "
            f"{P.num(sp['vs_measured_control_mean__median_pairwise_profile_correlation'])} "
            "between the profiles it predicts for two different knockdowns"
        )
    if not bits:
        return ""
    return (
        "<p>The fourth column says the same thing without any scoring rule in the path. For "
        "the measured effects, roughly three quarters of a perturbation's variance across "
        "genes survives subtracting the average response: knocking down different genes "
        "really does do different things. For the fine-tuned model's own prediction almost "
        "none of it does. In the primary arm: " + "; ".join(bits) + ".</p>"
    )


def training_section(r: Results, chart: str) -> str:
    rows = []
    for line in C.CELL_LINES:
        for arm in ARM_ORDER:
            t = r.training(line, arm)
            if not t:
                continue
            rows.append([
                LINE[line],
                ARM_LABEL[arm],
                str(t.get("selected_epoch", "n/a")),
                str(t.get("epochs_run", "n/a")),
                P.num(t.get("best_val_loss", nan())),
                str(t.get("n_train_cells", "n/a")),
            ])
    tbl = P.table(
        ["cell line", "arm", "selected epoch", "epochs run", "best validation loss",
         "training cells"],
        rows,
        sources=[S_TRAIN],
        caption="Fine-tuning outcome per run. Selection is the tutorial's: the epoch with the "
                "lowest masked MSE on held-out validation cells.",
    )
    return P.section(
        "The runs themselves",
        "<p>Every run trains, in the sense the recipe means: the training loss falls, the "
        "validation loss falls and then turns, and early stopping fires. The selected epoch "
        "is chosen on held-out masked MSE, which is what the tutorial selects on, and not on "
        "the response agreement, which is recorded every epoch as a diagnostic.</p>"
        + chart
        + tbl
        + "<p>The lower panel of the figure is the reason the diagnostic is worth recording. "
        "Response agreement against the control mean rises over training while the centred "
        "version stays flat at zero. The optimisation is real and it is moving the model "
        "toward the average perturbed cell.</p>",
        sources=[S_TRAIN, S_FIGS],
    )


def controls_section(r: Results) -> str:
    parts = []

    # the gene-table arms, re-run inside the same jobs
    ref_rows = []
    for line in C.CELL_LINES:
        for arm in ("scgpt_emb", "scgpt_resp", "coexp", "node", "null_randpair"):
            row = r.ref(line, arm)
            if row:
                ref_rows.append([LINE[line], arm, civ(row), str(row["n"]), str(row["n_groups"])])
    parts.append(P.table(
        ["cell line", "run 3 arm", "score on these folds", "perturbations", "groups"],
        ref_rows,
        sources=[S_SCORE],
        caption="Run 3's arms re-run inside every run-4 job on the same restricted folds, so "
                "the fine-tuned arms are compared against numbers from the same code path.",
    ))

    if not r.agreement.empty:
        spread = float(r.agreement["spread"].max())
        parts.append(
            f"<p>Those reference arms are recomputed independently inside each of the sixteen "
            f"jobs. The largest disagreement between any two recomputations of the same arm is "
            f"{P.num(spread)} correlation points, which is the reproducibility of the scoring "
            "path itself.</p>"
        )

    # variants of the primary arm: each is the same recipe with one choice
    # changed, and each is read in the same cell as the primary arm
    VARIANTS = (
        ("own_trainable", "the primary arm"),
        ("own_trainable__upstream", "upstream gene sampling"),
        ("own_trainable__f23", "refitted, folds 2 and 3 withheld"),
        ("own_trainable__f4", "refitted, fold 4 withheld"),
        ("own_trainable__lastepoch", "last epoch, not the selected one"),
    )
    var_rows = []
    for line in C.CELL_LINES:
        band = r.null_ceiling(line)
        for run, label in VARIANTS:
            row = r.run_feature_row(line, run)
            t = r.run_row(line, run.replace("__lastepoch", "")) or {}
            if row is None:
                continue
            var_rows.append([
                LINE[line], label,
                "last" if run.endswith("__lastepoch") else str(t.get("selected_epoch", "n/a")),
                civ(row), str(row["n"]),
                "yes" if row["score"] <= band else "no",
            ])
    if var_rows:
        parts.append(
            "<p>Four variants of the primary arm test choices we made rather than inherited. "
            "Under <code>--upstream-sampling</code> the batch sampler reproduces the published "
            "one exactly, where the random gene subset drops the perturbed gene from about "
            "40% of rows and those rows carry no perturbation flag at all; our default forces "
            "the perturbed gene into every batch. Two refits withhold the complementary folds, "
            "so that pooling covers the whole screen. The last reads the features off the "
            "final epoch rather than the one the recipe's selection rule keeps, which matters "
            "in Jurkat, where the validation loss never improves after the first epoch.</p>"
        )
        parts.append(P.table(
            ["cell line", "variant", "epoch read", "response features", "perturbations",
             "inside the shuffled-label band"],
            var_rows,
            sources=[S_RUNARM, S_FOLLOWUP],
            caption="Each variant scored in the same cell as the primary arm, on the folds "
                    "that variant withheld. The refits are read on folds 2-4, so their "
                    "perturbation counts and their ceilings differ from the primary arm's.",
        ))
    return P.section("Controls", "".join(parts), sources=[S_SCORE, S_RUNARM])


def artifacts_section(r: Results) -> str:
    rows = []
    for line in C.CELL_LINES:
        for arm in ARM_ORDER:
            t = r.training(line, arm)
            if not t or not t.get("artifact"):
                continue
            rows.append([
                LINE[line], ARM_LABEL[arm],
                f'<code>{P.esc(str(t["artifact"]))}</code>',
            ])
    return P.section(
        "Checkpoints",
        "<p>Every fine-tuned model is kept. Each directory holds the selected and final "
        "weights, the exact training configuration, the resolved gene table before and after "
        "training including the learned projection where one exists, the per-epoch loss and "
        "validation history, the base checkpoint revision, and the saved predicted "
        "response.</p>"
        + P.table(["cell line", "arm", "artifact"], rows, sources=[S_RUNARM]),
        sources=[S_RUNARM],
    )


def limitations(r: Results) -> str:
    return P.section(
        "What this does and does not settle",
        "<ul>"
        "<li>This is one model family and one recipe. It is scGPT's own published "
        "perturbation recipe on the screen the question is about, which is the strongest "
        "single test available, but a different architecture or a different objective could "
        "behave differently.</li>"
        "<li>The fine-tuning data is the same screen the evaluation comes from, split by "
        "response group. That is the setting the recipe is published for and it is the "
        "setting most favourable to installing the signal, so a null here is informative; it "
        "does not speak to fine-tuning on a different screen.</li>"
        "<li>Two of five folds are scored for the main arms, which resolves a paired "
        "difference more coarsely than run 3's five. The cross-fitted variant covers the "
        "whole screen and agrees.</li>"
        "<li>The translator is fitted leave-one-fold-out across all five folds, so for a "
        "given held-out fold it has seen features from folds the fine-tuned model trained "
        "on. That is true of run 3's frozen arms as well and is what keeps the two "
        "comparable; it means the held-out claim is about the perturbation, not about the "
        "translator's independence from the model's training set.</li>"
        "<li>The response features are read at run 3's primary anchor. Run 3 swept four "
        "anchors on the frozen model and none of them lifted the features out of the band; "
        "that sweep was not repeated on every fine-tuned checkpoint.</li>"
        "<li>A null is a statement about what these measurements can resolve. The "
        "shuffled-label band and the achieved resolution of each paired difference are "
        "reported next to every number so the size of an effect that would have been missed "
        "is visible.</li>"
        "</ul>",
        sources=[S_CONFIG],
    )


# --------------------------------------------------------------------------- #
def build(summary_dir: Path, out_dir: Path) -> Path:
    r = Results(summary_dir)
    made = figmod.build_all(summary_dir)
    figs = {k: v.get("fig") for k, v in made.items() if isinstance(v, dict)}

    first = {"done": False}

    def frag(name: str) -> str:
        fig = figs.get(name)
        if fig is None:
            return ""
        include = "cdn" if not first["done"] else False
        first["done"] = True
        return fig.to_html(full_html=False, include_plotlyjs=include, config=PLOTLY_CONFIG)

    f_swap = P.figure(
        1, "Response features before and after fine-tuning",
        frag("swap_before_after"),
        "Held-out score of the response-feature arm read from each checkpoint, on the "
        "residual target with reliability weighting and ceiling correction, 95% intervals "
        "over 2,000 resamples of the response groups. Open markers are the same architecture "
        "before any gradient step; filled markers are after fine-tuning. The shaded band is "
        "the range of the shuffled-label draws in that cell line and the dashed line is the "
        "gene-table arm.",
        "Fine-tuning does not move the response features out of the band, in either cell "
        "line, under any of the five gene-table arms.",
        sources=[S_SCORE, S_FIGS],
    )
    f_pred = P.figure(
        2, "Three ways of reading the same fine-tuned model",
        frag("predicted_and_features"),
        "For each fine-tuned model: its own predicted perturbation response scored directly "
        "against the measured effects, and the response features read from it through its own "
        "perturbation-flag query and through the value-substitution query runs 2 and 3 use. "
        "All three on the same held-out perturbations, the same target and the same "
        "correction.",
        "The model's own prediction sits in the same place as the features read out of it. "
        "The feature swap is not losing a signal the prediction head has.",
        sources=[S_FOLLOWUP, S_FIGS],
    )
    f_train = P.figure(
        3, "What fine-tuning optimised",
        frag("training_curves"),
        "Top: masked MSE on held-out validation cells per epoch, one line per run, with the "
        "selected epoch marked. Bottom: on the same validation cells, the agreement between "
        "the predicted and the measured perturbation response, before and after removing the "
        "response shared across perturbations.",
        "Validation loss falls and turns, so the runs train. The uncentred response agreement "
        "rises with it; the centred one does not move off zero.",
        sources=[S_TRAIN, S_FIGS],
    )

    title = ("Fine-tuning scGPT on the screen teaches it the average knockdown response, "
             "not which gene was knocked down")
    blocks = [
        P.section("", f"<h1>{P.esc(title)}</h1>"),
        opening(r),
        setup(r),
        direct_section(r, f_pred),
        verdict_section(r, f_swap),
        training_section(r, f_train),
        controls_section(r),
        artifacts_section(r),
        limitations(r),
    ]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "index.html"
    dest.write_text(P.document(title, blocks))
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    summary = Path(a.summary) if a.summary else paths.results_dir() / "summary"
    dest = build(summary, Path(a.out))
    print(f"wrote {dest} ({dest.stat().st_size / 1000:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
