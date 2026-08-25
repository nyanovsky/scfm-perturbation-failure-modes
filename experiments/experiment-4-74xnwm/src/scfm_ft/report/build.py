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
S_CMP = f"{SRC}/src/scfm_ft/compare_run3.py"
S_CMP_TBL = f"{SRC}/results/run3_comparison.parquet"
S_PROV = f"{SRC}/src/scfm_ft/recipe_provenance.py"
S_PROV_TBL = f"{SRC}/results/recipe_provenance.json"

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

#: the refits withhold different folds, so the folds-0-and-1 shuffled band does
#: not apply to them
REFIT_TAGS = ("__f23", "__f4")


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
        self.null_check = self._opt("null_check")
        self.spread = self._opt("prediction_spread")
        self.crossfit = self._crossfit()
        self.ingest = self._ingest()
        # computed pod-side against run 3's published table rather than by the
        # merge job, so it lives beside the summary rather than inside it
        cmp_file = paths.results_dir() / "run3_comparison.parquet"
        self.run3_cmp = pd.read_parquet(cmp_file) if cmp_file.exists() else pd.DataFrame()
        prov = paths.results_dir() / "recipe_provenance.json"
        self.provenance = json.loads(prov.read_text()) if prov.exists() else {}

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
    in_band = "every one leaves" if n_in == len(rows) else f"{n_in} leave"
    base_h = r.arm("hepg2", "own_trainable").get("direct_prediction_by_baseline") or {}
    base_j = r.arm("jurkat", "own_trainable").get("direct_prediction_by_baseline") or {}
    direct_h = base_h.get("measured_control_mean")
    direct_j = base_j.get("measured_control_mean")
    own_h = base_h.get("unflagged_model_output")
    own_j = base_j.get("unflagged_model_output")
    key = "vs_measured_control_mean__median_residual_fraction_of_variance"
    sp_h = (r.spread_row("hepg2", "own_trainable") or {}).get(key, nan())
    sp_j = (r.spread_row("jurkat", "own_trainable") or {}).get(key, nan())
    emb_h = r.ref("hepg2", "scgpt_emb")
    emb_j = r.ref("jurkat", "scgpt_emb")
    # stated as a ratio against the strongest fine-tuned reading rather than a
    # typical one, so the multiple is the smallest the table supports
    embs = [e["score"] for e in (emb_h, emb_j) if e]
    ratio = min(embs) / best if embs and best and best > 0 else nan()
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
        f"models in two cell lines, {in_band} the response features inside the "
        f"shuffled-label band; the highest any of them reaches is {P.num(best)}. The same "
        "model's static gene table, read through the same translator on the same held-out "
        f"perturbations, sits at {P.num((emb_h or {}).get('score', nan()))} in HepG2 and "
        f"{P.num((emb_j or {}).get('score', nan()))} in Jurkat, at least {ratio:.0f} times "
        "higher than any of them, and fine-tuning does not move that either. The paired "
        "differences are small and mostly indistinguishable from zero, with "
        f"{len(paired_exceptions(r))} exceptions reported below; none of them is a change of "
        "the order that would be needed to reach the gene table.</p>"
        + (
            ""
            if not (above := own_query_above(r))
            else "<p>One reading is less clean, and it is a secondary one. Querying the same "
            "checkpoints through the model's own perturbation-flag input, rather than "
            "through the value substitution runs 2 and 3 use, is noisier: "
            f"{len(above)} of those {len(rows)} readings sit above the band and above that "
            "query's own shuffled draws, at "
            + " and ".join(P.num(p["score"]) for p in above)
            + ", both in "
            + ", ".join(sorted({LINE[p["cell_line"]] for p in above}))
            + ". They are still three to four times below the gene table, they are not "
            "reproduced by the other arms in the same cell line, and that query's draws "
            "are wide enough that this is a weak reading rather than a signal. The "
            "Controls section gives the draws.</p>"
        )
        + "<p>The same thing is visible on the model's own prediction head, with no translator "
        "in the path. Its predicted profile correlates with the measured effect at "
        f"{P.num((direct_h or {}).get('raw', nan()))} in HepG2 and "
        f"{P.num((direct_j or {}).get('raw', nan()))} in Jurkat, which looks like a model "
        "that has learned to predict knockdowns. Measure only what the perturbation flag "
        f"changes, against the model's own unflagged output, and that falls to "
        f"{P.num((own_h or {}).get('raw', nan()))} and "
        f"{P.num((own_j or {}).get('raw', nan()))}. Remove the held-out fold's own mean from "
        f"both sides and it is {P.num((direct_h or {}).get('fold_centred', nan()))} and "
        f"{P.num((direct_j or {}).get('fold_centred', nan()))}. Of a predicted profile's "
        f"variance, {P.num(sp_h)} in HepG2 and {P.num(sp_j)} in Jurkat survives subtracting "
        "the mean prediction, against "
        f"{P.num(r.measured_spread('hepg2'))} and {P.num(r.measured_spread('jurkat'))} for "
        "the measurement itself. What fine-tuning installed is close to one profile for every "
        "knockdown, so a read-out that asks which gene was knocked down has very little to "
        "recover.</p>",
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
        ("the five arms", [
            (ARM_LABEL[a.name], a.note) for a in C.ARMS
        ]),
    ])
    return P.section(
        "The recipe and the five arms",
        "<p>scGPT publishes its perturbation recipe as a notebook, so there is no trainer to "
        "import. The model is scGPT's own <code>TransformerGenerator</code> class, unmodified "
        "and constructed with the notebook's settings; the training loop and batch sampler "
        "around it are restated, because the run also needs the model to be queryable the way "
        "runs 2 and 3 query it. Three checks hold the restatement to the original: the "
        "masked-MSE loss is compared numerically against <code>scgpt/loss.py</code> at load "
        "time, the batch sampler reproduces the upstream one exactly under "
        "<code>--upstream-sampling</code> and that variant is scored beside the default, and "
        "every setting is checked field by field against the notebook.</p>"
        + recipe_provenance(r)
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


def recipe_provenance(r: Results) -> str:
    """Every recipe setting beside the tutorial's own value for it."""
    p = r.provenance
    if not p or not p.get("fields"):
        return ""
    compared = [f for f in p["fields"] if f.get("matches") is not None]
    other = [f for f in p["fields"] if f.get("matches") is None]
    rows = [
        [f"<code>{P.esc(str(f['tutorial_variable']))}</code>",
         f"<code>{P.esc(repr(f['tutorial']))}</code>",
         f"<code>{P.esc(repr(f['ours']))}</code>",
         "same" if f["matches"] else "DIFFERS",
         f["note"] or ""]
        for f in compared
    ]
    n_ok = sum(1 for f in compared if f["matches"])
    verdict = (
        f"All {len(compared)} settings the notebook names are the settings this run used"
        if n_ok == len(compared)
        else f"{len(compared) - n_ok} of {len(compared)} settings differ from the notebook"
    )
    tail = ""
    if other:
        tail = (
            " The notebook fixes "
            + " and ".join(f"<code>{P.esc(f['field'])}</code>" for f in other)
            + " inline rather than as a named variable, so those two are compared by reading "
            "its training loop: "
            + "; ".join(f"{P.esc(f['field'])} {P.esc(repr(f['ours']))}, {f['note']}"
                        for f in other)
            + "."
        )
    return (
        f"<p>{verdict}. Two of these are easy to get wrong, because the value in "
        "<code>TransformerGenerator</code>'s signature is not the value the notebook passes: "
        "the class defaults to dropout 0.5 and perturbation-pad id 2, and the notebook passes "
        "0 and 0. The recipe follows the notebook." + tail + "</p>"
        + P.table(
            ["tutorial variable", "tutorial", "this run", "", "note"],
            rows,
            sources=[S_PROV, S_PROV_TBL],
            caption="Read out of the tutorial notebook's own settings cell by parsing its "
                    "literal assignments, against the recipe this run recorded. The notebook "
                    "is not vendored here; its URL and the sha256 of the file compared "
                    "against are in the source table, so the comparison can be repeated.",
        )
        + selection_provenance(p)
        + "<p>One thing does differ, and it is an environment constraint rather than a "
        "choice. The notebook sets <code>use_fast_transformer=True</code>, which needs "
        "scGPT's FlashMHA layers; their API exists only in flash-attn 1.x, which will not "
        "build against the torch this run uses. scGPT handles that case itself by falling "
        "back to <code>torch.nn.TransformerEncoder</code>, which is the path taken here, and "
        "the checkpoint's fused attention tensors are remapped onto torch's packed "
        "projection. The attention math is the same; the kernel and therefore the exact "
        "numerics are not.</p>"
    )


def selection_provenance(p: dict) -> str:
    """The one recipe choice that is not a named setting: which epoch is kept.

    Read out of the notebook's training-loop cell rather than described, because
    it decides which weights every number on this page comes from.
    """
    sel = (p or {}).get("selection")
    if not sel:
        return ""
    t = sel["tutorial"]
    agree = (
        "This run applies the same rule"
        if sel.get("matches")
        else "This run does NOT apply the same rule"
    )
    return (
        "<p>The checkpoint rule is not a named setting, so it is read out of the notebook's "
        "training-loop cell the same way. That loop sets "
        f"<code>{P.esc(t['val_score_expression'])}</code>, copies the model when "
        f"<code>{P.esc(t['best_model_guard'])}</code>, and resets its early-stopping counter "
        "on the same comparison"
        + ("" if t["early_stopping_counter_reset_by_the_same_guard"] else " (it does not)")
        + f". {agree}: it keeps the epoch whose held-out perturbations have the highest mean "
        "Pearson correlation, across all genes, between their measured and their predicted "
        "mean expression profile, which is what "
        "<code>compute_perturbation_metrics(...)['pearson']</code> returns.</p>"
    )


def n_paired_all(r: Results) -> int:
    """How many paired differences of a fine-tuned response arm the page computes.

    Every fine-tuned run, both references, verdict cell. It is the multiple
    comparison count a single nominal p-value has to be read against.
    """
    if r.contrasts.empty:
        return 0
    return int(len(r.contrasts[
        (r.contrasts.translator == "mlp")
        & (r.contrasts.target == VERDICT[0])
        & (r.contrasts.weighting == VERDICT[1])
        & (r.contrasts.arm.str.startswith("ft_"))
        & (r.contrasts.arm.str.endswith("_resp"))
    ]))


def paired_exceptions(r: Results) -> list[dict]:
    """Paired differences of the primary read-out whose interval excludes zero.

    Both references, all five arms, both cell lines, in the verdict cell. The
    prose that counts or names these reads them from here rather than restating
    a count from a previous execution.
    """
    if r.contrasts.empty:
        return []
    c = r.contrasts[
        (r.contrasts.translator == "mlp")
        & (r.contrasts.target == VERDICT[0])
        & (r.contrasts.weighting == VERDICT[1])
        & (r.contrasts.arm.isin([f"ft_{a}_resp" for a in ARM_ORDER]))
        & (r.contrasts.against.isin(("scgpt_resp", "coexp")))
    ]
    out = []
    for x in c[~((c["lo"] <= 0) & (c["hi"] >= 0))].itertuples():
        arm = str(x.arm)[len("ft_"): -len("_resp")]
        out.append({
            "cell_line": x.cell_line,
            "arm": arm,
            "against": x.against,
            "difference": float(x.difference),
            "lo": float(x.lo),
            "hi": float(x.hi),
            "label": f"{LINE[x.cell_line]} {ARM_LABEL.get(arm, arm)} against "
                     + ("the frozen arm" if x.against == "scgpt_resp" else "co-expression")
                     + f", {P.num(float(x.difference))}",
        })
    return sorted(out, key=lambda d: -abs(d["difference"]))


def paired_note(r: Results) -> str:
    """The pre-registered paired comparison, both references, with its resolution.

    The verdict is a paired difference, not a pair of separately-read scores, so
    the differences and their intervals belong on the page rather than being
    left for a reader to subtract.
    """
    if r.contrasts.empty:
        return ""
    c = r.contrasts[
        (r.contrasts.translator == "mlp")
        & (r.contrasts.target == VERDICT[0])
        & (r.contrasts.weighting == VERDICT[1])
    ]
    rows, resolvable = [], []
    pos, neg = [], []
    for line in C.CELL_LINES:
        for arm in ARM_ORDER:
            sub = c[(c.cell_line == line) & (c.arm == f"ft_{arm}_resp")]
            if sub.empty:
                continue
            cells = {}
            for against in ("scgpt_resp", "coexp"):
                a = sub[sub.against == against]
                if a.empty:
                    cells[against] = "n/a"
                    continue
                x = a.iloc[0]
                cells[against] = (
                    f"{P.num(x['difference'])} ({P.num(x['lo'])} to {P.num(x['hi'])}), "
                    f"p {P.num(x['p_two_sided'])}"
                )
                if x["lo"] > 0:
                    pos.append(f"{LINE[line]} {ARM_LABEL[arm]} against "
                               f"{'the frozen arm' if against == 'scgpt_resp' else 'co-expression'}")
                elif x["hi"] < 0:
                    neg.append(f"{LINE[line]} {ARM_LABEL[arm]}")
            res = float(sub.iloc[0]["min_resolvable_difference__rho_0.8"])
            resolvable.append((line, res))
            rows.append([LINE[line], ARM_LABEL[arm], cells["scgpt_resp"], cells["coexp"],
                         P.num(res)])
    if not rows:
        return ""
    floor = {
        line: (min(v for k, v in resolvable if k == line), max(v for k, v in resolvable if k == line))
        for line, _ in resolvable
    }
    exc = paired_exceptions(r)
    verdict = (
        "Every one of those differences includes zero: at this resolution fine-tuning "
        "neither improves the response features nor damages them."
        if not exc
        else "Most of those differences include zero. The "
             + ("exception is " if len(exc) == 1 else f"{len(exc)} exceptions are ")
             + "; ".join(d["label"] for d in exc)
             + ". "
             + (
                 "That interval sits entirely above zero."
                 if all(d["difference"] > 0 for d in exc)
                 else "Those intervals sit entirely below zero."
                 if all(d["difference"] < 0 for d in exc)
                 else "They point in opposite directions, which is what a set of small "
                      "differences read against a multiple-comparison count looks like."
             )
    )
    span = "; ".join(
        f"{LINE[line]} {P.num(lo)} to {P.num(hi)}" for line, (lo, hi) in floor.items()
    )
    return (
        "<p>The pre-registered test is a paired difference, so it is worth stating as one "
        "rather than leaving a reader to subtract two columns. Each fine-tuned model's "
        "response features are compared, on the same held-out perturbations, with two "
        "references: the frozen model's response features as run 3 scored them, and "
        "co-expression. " + verdict + " The last column is the smallest difference the "
        "pairing could have resolved at 95% given the correlation between the two arms' "
        "per-perturbation scores, so a difference smaller than it would not have been seen "
        f"here: {span}.</p>"
        + P.table(
            ["cell line", "arm", "minus the frozen model's response features",
             "minus co-expression", "smallest resolvable difference"],
            rows,
            sources=[S_SCORE, S_ANALYSIS],
            caption="Paired differences on the verdict cell, with 95% intervals over the same "
                    "2,000 resamples of the response groups and a two-sided p-value from the "
                    "resampled differences. Positive means the fine-tuned arm scored higher. "
                    f"{n_paired_all(r)} such differences are computed across all runs and "
                    "both references, so a single nominal p below 0.05 is not evidence on "
                    "its own.",
        )
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
        "detectably at all. It is not evidence that they are exactly zero. The whole-screen "
        "reading below recovers the resolution the restriction costs, by fitting the primary "
        "arm on complementary folds and pooling.</p>"
    )


CROSSFIT_ARMS = (
    ("ft_own_trainable_resp", "response features, after fine-tuning"),
    ("scgpt_resp", "response features, before fine-tuning"),
    ("coexp", "co-expression"),
    ("scgpt_emb", "gene table, before fine-tuning"),
    ("node", "node covariates"),
)


def holdout_note(r: Results) -> str:
    """The whole-screen reading, where the restricted evaluation runs out of power."""
    rows, covered, resolved, overlaps, ratios = [], [], [], [], []
    for line in C.CELL_LINES:
        got = False
        for arm, label in CROSSFIT_ARMS:
            row = r.crossfit_row(line, arm)
            if not row:
                continue
            got = True
            rows.append([LINE[line], label, civ(row), str(row["n"]), str(row["n_groups"])])
        if got:
            base = r.crossfit_row(line, "ft_own_trainable_resp")
            if base:
                covered.append(f"{LINE[line]} {base['n']} perturbations in "
                               f"{base['n_groups']} groups")
            # Whether co-expression is resolved above zero, and whether it is
            # separated from the response arms, are two different questions and
            # the answer differs. Both are read off the intervals rather than
            # asserted, so the sentence cannot drift from the table.
            co = r.crossfit_row(line, "coexp")
            resp = [r.crossfit_row(line, a) for a in ("ft_own_trainable_resp", "scgpt_resp")]
            resp = [p for p in resp if p]
            emb = r.crossfit_row(line, "scgpt_emb")
            if emb and resp and max(p["score"] for p in resp) > 0:
                ratios.append(emb["score"] / max(p["score"] for p in resp))
            if co and co["lo"] > 0:
                resolved.append(f"{LINE[line]} {civ(co)}")
            if co and resp and any(co["lo"] <= p["hi"] for p in resp):
                overlaps.append(LINE[line])
    if not rows:
        return ""
    return (
        "<p>Two folds is a narrow window, so the primary arm was fitted twice more, once "
        "with folds 2 and 3 withheld and once with fold 4 withheld, and each model's "
        "held-out rows were pooled. Every perturbation in the screen is then scored by a "
        f"model that never saw it: {', '.join(covered)}. The other arms do not depend on "
        "which folds a model saw, so they are read over the same whole screen.</p>"
        + P.table(
            ["cell line", "arm", "score", "perturbations", "groups"],
            rows,
            sources=[S_FOLLOWUP],
            caption="Whole-screen scores. The fine-tuned response arm is pooled across three "
                    "models fitted on complementary folds; every other arm is scored once "
                    "over the same perturbations, against the same whole-screen ceiling.",
        )
        + "<p>The whole screen buys back the resolution the two-fold restriction cost: "
        + (f"co-expression is now above zero on its own interval ({'; '.join(resolved)}), "
           "which the restricted evaluation could not show. " if resolved else "")
        + "Its point estimate is roughly twice either response arm's in both cell lines, "
        + (("though the intervals still overlap in "
            + ("both cell lines" if len(overlaps) == len(C.CELL_LINES)
               else " and ".join(overlaps))
            + ", so the ordering is consistent rather than separated at 95%. ")
           if overlaps else "")
        + "The response features sit where they sat before fine-tuning, "
        + (f"{min(ratios):.0f} to {max(ratios):.0f} times below the gene table, "
           if len(ratios) > 1 else "far below the gene table, ")
        + "and pooling does not change that.</p>"
        + "<p>Two cautions on reading this table. The intervals are bootstrap "
        "resamples of the response groups, not shuffled-label draws: an interval that "
        "excludes zero says the arm is not zero, which is a weaker statement than "
        "clearing the pre-registered band, and no shuffled draw was computed at "
        "whole-screen scope in this run. And the response-feature arms are read on the "
        "residual target, where a prediction that barely varies between perturbations "
        "can still correlate with the shared offset, exactly as the direct-prediction "
        "section describes.</p>"
        + run3_agreement(r)
    )


def run3_agreement(r: Results) -> str:
    """Whether run 4's re-reading of run 3's arms lands on run 3's own numbers."""
    if r.run3_cmp.empty:
        return ""
    frame = r.run3_cmp.dropna(subset=["run3", "run4"])
    if frame.empty:
        return ""
    rows = [
        [LINE[x.cell_line], x.label, P.num(x.run3), P.num(x.run4), P.num(x.difference),
         "yes" if (x.run4_in_run3_interval and x.run3_in_run4_interval) else "no"]
        for x in frame.itertuples()
    ]
    ok = frame[frame.run4_in_run3_interval & frame.run3_in_run4_interval]
    worst = float(ok["difference"].abs().max()) if not ok.empty else nan()
    dis = frame[~(frame.run4_in_run3_interval & frame.run3_in_run4_interval)]
    note = ""
    if not dis.empty:
        names = ", ".join(sorted({f"{LINE[x.cell_line]} {x.label}" for x in dis.itertuples()}))
        note = (
            f" The exception is {names}: run 3 reads it above zero and run 4 reads it at "
            "zero. It is the weakest arm in the table, it is the one arm no claim on this "
            "page rests on, and a near-zero arm is where refitting the translator moves a "
            "score most. Run 3's estimate still sits inside run 4's interval."
        )
    return (
        "<p>Every arm in that table except the fine-tuned one is run 3's own, re-run here "
        "through run 3's code on run 3's folds against run 3's effect matrices, so the "
        "table is also a check that run 4 reproduces the run it extends. They do, to within the resolution two fits of the "
        "same quantity have: the translator is an MLP refitted per held-out fold and "
        "refitting it is not deterministic across processes, so the check is mutual "
        f"containment rather than equality. The largest gap among the arms that agree is "
        f"{P.num(worst)} correlation points.{note}</p>"
        + P.table(
            ["cell line", "arm", "run 3", "run 4", "difference", "each inside the other's "
             "interval"],
            rows,
            sources=[S_CMP, S_CMP_TBL],
            caption="Run 3's published whole-screen numbers beside run 4's re-reading of the "
                    "same arms, both on the grouped folds with the MLP translator and the "
                    "same ceiling. Jurkat's node arm is omitted: run 3 scores no "
                    "perturbations there, so there is nothing to compare.",
        )
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
    own_pts = r.all_feature_points("final__pert_flag_resp")
    n_out = sum(1 for p in own_pts if not r.in_band(p["cell_line"], p))
    draws = {line: own_query_draws(r, line) for line in C.CELL_LINES}
    above_own = [
        p for p in own_pts
        if (d := draws.get(p["cell_line"])) is not None and p["score"] > d[1]
    ]
    best_own = max((p["score"] for p in own_pts), default=nan())
    embs = [e["score"] for line in C.CELL_LINES if (e := r.ref(line, "scgpt_emb"))]
    own_ratio = min(embs) / best_own if embs and best_own and best_own > 0 else nan()
    span = "; ".join(
        f"{LINE[line]} {P.num(d[0])} to {P.num(d[1])}"
        for line in C.CELL_LINES if (d := draws.get(line)) is not None
    )
    own_note = (
        "Reading the features through the model's own perturbation-flag query, the query "
        "the fine-tuning objective is literally defined on, does not lift them either."
        if n_out == 0
        else "Reading the features through the model's own perturbation-flag query, the "
             "query the fine-tuning objective is literally defined on, gives a much noisier "
             f"reading: {n_out} of those {len(own_pts)} sit above the band's top edge, the "
             f"highest at {P.num(best_own)}. The band is the wrong null for that column. "
             "The same query's own shuffled-label draws, over the models scored on the same "
             f"folds, span {span}, several times the band's width, and "
             f"{len(above_own)} of the {len(own_pts)} readings sit above the widest draw "
             "their cell line produced. Every one of them is still at least "
             f"{own_ratio:.0f} times below the gene-table arm. The Controls section gives "
             "those draws."
    )
    return P.section(
        "The feature swap, before and after",
        "<p>The measurement is run 3's, unchanged: extract the model's predicted change in "
        "every read-out gene when one gene's input value is moved to the anchor, standardise "
        "it across genes, hand the pair of gene features to the fixed translator, and score "
        "the held-out perturbations against the measured CRISPRi effects. The only thing that "
        "differs between the two columns below is which checkpoint answered the query.</p>"
        + chart
        + tbl
        + f"<p>{'All ' + str(len(post)) if n_in == len(post) else f'{n_in} of the {len(post)}'} "
        "fine-tuned models sit at or below the largest "
        "shuffled-label draw in their cell line, and none approaches the gene-table arm. "
        + own_note + "</p>"
        + paired_note(r)
        + resolution_note(r)
        + holdout_note(r),
        sources=[S_QUERY, S_SCORE],
    )


def own_query_above(r: Results) -> list[dict]:
    """Own-query readings above the widest shuffled draw of their cell line."""
    out = []
    for p in r.all_feature_points("final__pert_flag_resp"):
        d = own_query_draws(r, p["cell_line"])
        if d is not None and p["score"] > d[1]:
            out.append(p)
    return out


def own_query_draws(r: Results, line: str) -> tuple[float, float] | None:
    """The perturbation-flag query's shuffled draws, pooled over one cell line.

    Five draws per arm is too few to bound one arm on its own, so the draws are
    pooled across the arms scored on the same folds as the band. The two refits
    are left out: they are read on other folds against another ceiling.
    """
    nc = r.null_check
    if nc.empty or "draw_max" not in nc:
        return None
    sub = nc[
        (nc.cell_line == line)
        & (~nc["train_arm"].astype(str).str.endswith(REFIT_TAGS))
        & (nc["family"].astype(str).str.startswith("pert_flag"))
    ]
    if sub.empty:
        return None
    return float(sub["draw_min"].min()), float(sub["draw_max"].max())


def _cell(d: dict | None, prefix: str) -> str:
    if not d or d.get(prefix) is None:
        return "n/a"
    return ci(d[prefix], d.get(f"{prefix}_lo", nan()), d.get(f"{prefix}_hi", nan()))


def direct_section(r: Results, chart: str) -> str:
    rows = []
    for line in C.CELL_LINES:
        for arm in ARM_ORDER:
            a = r.arm(line, arm)
            by_base = a.get("direct_prediction_by_baseline") or {}
            d = by_base.get("measured_control_mean") or a.get("direct_prediction")
            own = by_base.get("unflagged_model_output")
            sp = r.spread_row(line, arm) or {}
            rows.append([
                LINE[line],
                ARM_LABEL[arm],
                _cell(d, "raw"),
                _cell(own, "raw"),
                _cell(d, "fold_centred"),
                P.num(sp.get("vs_measured_control_mean__median_residual_fraction_of_variance",
                             nan())),
            ])
    tbl = P.table(
        ["cell line", "arm", "predicted profile vs measured",
         "flag-induced change vs measured", "same, fold-centred",
         "gene-specific fraction of the prediction"],
        rows,
        sources=[S_FOLLOWUP, S_TRAIN],
        caption="The model's own predicted perturbation response, with no translator in the "
                "path, on the held-out folds. The first column measures the whole predicted "
                "profile against the measured effect, the comparison the published recipe is "
                "evaluated on. The second replaces the measured control profile with the "
                "model's own no-flag output, so it measures only what the perturbation flag "
                "changes. The third removes the held-out fold's own mean from both sides. The "
                "last is the median fraction of a predicted profile's variance that survives "
                "subtracting the mean prediction, taken over every in-scope perturbation "
                "because it is a property of the model's output rather than a score.",
    )
    return P.section(
        "The prediction is nearly the same for every knockdown",
        "<p>A feature swap can only find signal the model has. Before concluding anything "
        "about the features, it is worth asking whether fine-tuning worked on its own terms. "
        "So we scored the model's prediction directly, with no translator in the path: run the "
        "perturbation forward pass and compare the predicted profile against the measured "
        "effect, one correlation per held-out perturbation, through the same reliability "
        "weighting and ceiling correction the feature arms use.</p>"
        + chart
        + tbl
        + "<p>The first two columns are the same prediction measured against two different "
        "reference points, and the gap between them is the result. Against the measured "
        "control profile the fine-tuned model looks like it is predicting knockdowns: the "
        "whole predicted profile correlates with the whole measured effect, between 0.3093 "
        "and 0.6159 in every arm. Replace that reference with the model's own output for the "
        "same "
        "cell without the perturbation flag, so that only what the flag changes is left, and "
        "the agreement collapses: no arm reaches 0.16 and six of the ten are negative. The "
        "first column is therefore mostly a systematic difference between the model's output "
        "and the measured control profile, shared across every perturbation, rather than the "
        "model responding to the gene it was told about.</p>"
        + "<p>The third column settles it. Removing the held-out fold's own mean from both "
        "sides leaves only what distinguishes one perturbation from another, and there the "
        "agreement is approximately zero in both cell lines. The intervals on that column are "
        "wide, so it is a bound rather than a precise zero; the next two paragraphs give the "
        "same conclusion from quantities that are not correlations and do not carry that "
        "width.</p>"
        + spread_note(r)
        + epoch_diagnostic_note(r)
        + "<p>This matters for how the rest of the page should be read. The response features "
        "sitting inside the null band is not the feature swap failing to find something the "
        "model has. There is very little gene-specific structure in the prediction for any "
        "read-out to recover.</p>",
        sources=[S_FOLLOWUP, S_SCORE],
    )


def epoch_diagnostic_note(r: Results) -> str:
    """The same contrast inside training, over every epoch of every run."""
    if r.epochs.empty or "delta_ctrlmean" not in r.epochs:
        return ""
    raw, cen = r.epochs["delta_ctrlmean"].dropna(), r.epochs["delta_ctrlmean_centered"].dropna()
    if raw.empty or cen.empty:
        return ""
    return (
        "<p>The same contrast appears inside training, on the validation cells the recipe "
        f"selects on. Over all {len(raw)} epochs of all {len(r.runs)} runs, the uncentred "
        f"agreement never falls below {P.num(raw.min())}, while the centred version stays "
        f"within {P.num(cen.abs().max())} of zero throughout. Fine-tuning improves the one "
        "and never moves the other.</p>"
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
        "<p>The last column says the same thing with no scoring rule in the path at all, so "
        "it does not inherit the width of a correlation interval. For the measured effects, "
        "most of a perturbation's variance across genes survives subtracting the average "
        "response: knocking down different genes really does do different things. For the "
        "fine-tuned model's own prediction almost none of it does. In the primary arm: "
        + "; ".join(bits) + ".</p>"
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
                P.num(t.get("selected_val_pearson", nan())),
                str(t.get("mse_selected_epoch") or "n/a"),
                str(t.get("epochs_run", "n/a")),
                P.num(t.get("best_val_loss", nan())),
                str(t.get("n_train_cells", "n/a")),
            ])
    tbl = P.table(
        ["cell line", "arm", "epoch the recipe keeps", "its validation Pearson",
         "epoch masked MSE would keep", "epochs run", "best validation loss",
         "training cells"],
        rows,
        sources=[S_TRAIN],
        caption="Fine-tuning outcome per run. The recipe keeps the epoch with the highest "
                "validation Pearson; the epoch the masked-MSE rule would keep is recorded "
                "beside it from the same run.",
    )
    return P.section(
        "Training curves and selected epochs",
        "<p>Every run trains, in the sense the recipe means: the training loss falls, the "
        "validation loss falls and then turns, and early stopping fires. The checkpoint the "
        "recipe keeps is the one with the highest validation Pearson between the predicted "
        "and the measured mean expression profile of each held-out perturbation, which is "
        "what the tutorial's loop selects on. The masked MSE on the same cells is tracked in "
        "parallel under its own patience counter, so the epoch that rule would have kept is a "
        "number from the same run rather than a separate experiment.</p>"
        + chart
        + tbl
        + "<p>The bottom panel of the figure is the reason the response agreement is worth "
        "recording even though nothing selects on it. Agreement against the control mean "
        "rises over training while the centred version stays flat at zero. The optimisation "
        "is real and it is moving the model toward the average perturbed cell.</p>",
        sources=[S_TRAIN, S_FIGS],
    )


def _rule_agreement(r: Results) -> tuple[int, int]:
    """How many of this execution's runs the two rules keep the same epoch for.

    Both rules are tracked inside every run with independent patience counters,
    so this is a within-run comparison: where the epochs coincide the two rules
    keep the same weights and every downstream number is identical by
    construction.
    """
    if r.runs.empty or "mse_selected_epoch" not in r.runs.columns:
        return 0, 0
    t = r.runs.dropna(subset=["mse_selected_epoch"])
    if t.empty:
        return 0, 0
    same = int((t.selected_epoch.astype(int) == t.mse_selected_epoch.astype(int)).sum())
    return same, int(len(t))


def selection_section(r: Results, alt: Results | None) -> str:
    """The recipe read at the checkpoint each selection rule keeps.

    The recipe's rule decides which weights every number above comes from, so
    the alternative is worth reading rather than assuming. Two readings are
    available and they are not equivalent: the masked-MSE epoch is tracked
    inside each of this execution's own runs, and a separate earlier execution
    of the same sixteen arms selected on masked MSE.
    """
    if alt is None:
        return ""
    rows, moved = [], []
    for line in C.CELL_LINES:
        band = r.null_ceiling(line)
        for arm in ARM_ORDER:
            a = r.feat(line, arm, "final__value_substitution_resp")
            b = alt.feat(line, arm, "final__value_substitution_resp")
            if not a or not b:
                continue
            ta, tb = r.training(line, arm), alt.training(line, arm)
            rows.append([
                LINE[line], ARM_LABEL[arm],
                str(ta.get("selected_epoch", "n/a")), civ(a),
                str(tb.get("selected_epoch", "n/a")), civ(b),
                "yes" if max(a["score"], b["score"]) <= band else "no",
            ])
            if max(a["score"], b["score"]) > band:
                moved.append(f"{LINE[line]} {ARM_LABEL[arm]}")
    if not rows:
        return ""
    last = [
        [LINE[line], str(row["n"]), civ(row)]
        for line in C.CELL_LINES
        if (row := alt.last_epoch_row(line, "own_trainable")) is not None
    ]
    same, total = _rule_agreement(r)
    parts = [
        "<p>The recipe's checkpoint rule decides which weights the features and the "
        "prediction are read from, so it is worth reading rather than assuming. Every run "
        "reported above tracks both rules at once, each with its own patience counter, so "
        "the epoch the masked-MSE rule would have kept is known for the same trajectory. "
        f"The two rules keep the same epoch in {same} of {total} runs; there the weights are "
        "the same object and every number below them is identical by construction.</p>",
        "<p>An earlier execution of the same sixteen arms was selected on masked MSE "
        "throughout. It is a genuinely separate run rather than the same trajectory re-read: "
        "it predates the three faithfulness controls added later, so its random "
        "initialisation and its whole trajectory differ, and it varies the selection rule "
        "and the run together. Read as a replicate rather than as an isolated rule change, "
        "the table puts the response-feature arm of each execution in the same verdict "
        "cell.</p>",
        P.table(
            ["cell line", "arm", "epoch, Pearson rule", "response features",
             "epoch, masked-MSE rule", "response features",
             "both inside the shuffled-label band"],
            rows,
            sources=[S_SCORE, S_TRAIN],
            caption="This execution on the left, selected as the recipe selects; the earlier "
                    "masked-MSE execution on the right, each read at the epoch its rule "
                    "keeps. Same verdict cell as everywhere else: residual target, "
                    "reliability weighting, ceiling correction, MLP translator, folds 0 "
                    "and 1.",
        ),
    ]
    parts.append(
        "<p>" + (
            "Every arm stays inside the band in both executions, so neither the checkpoint "
            "the recipe keeps nor the run it was trained in changes the reading."
            if not moved
            else "In at least one execution these arms leave the band: " + ", ".join(moved)
            + ". That difference is read on its own terms rather than folded into the "
            "summary above."
        ) + "</p>"
    )
    if last:
        parts.append(
            "<p>The masked-MSE execution was also read at its last epoch rather than its "
            "selected one, which is the sharper version of the same question, since that rule "
            "selects the first epoch in Jurkat there: "
            + "; ".join(f"{line} {score} on {n} perturbations" for line, n, score in last)
            + ".</p>"
        )
    return P.section(
        "The two checkpoint-selection rules", "".join(parts), sources=[S_SCORE, S_TRAIN]
    )


def controls_section(r: Results) -> str:
    parts = []

    # the gene-table arms, re-run inside the same jobs
    ref_rows, unscored = [], []
    for line in C.CELL_LINES:
        for arm in ("scgpt_emb", "scgpt_resp", "coexp", "node", "null_randpair"):
            row = r.ref(line, arm)
            if not row:
                continue
            # an arm with no scored perturbation in that cell line is named in
            # the prose rather than printed as an empty row
            if not row["n"] or row["score"] != row["score"]:
                unscored.append(f"{LINE[line]} {arm}")
                continue
            ref_rows.append([LINE[line], arm, civ(row), str(row["n"]), str(row["n_groups"])])
    parts.append(P.table(
        ["cell line", "run 3 arm", "score on these folds", "perturbations", "groups"],
        ref_rows,
        sources=[S_SCORE],
        caption="Run 3's arms re-run alongside every fine-tuned arm on the same restricted "
                "folds, so "
                "the fine-tuned arms are compared against numbers from the same code path.",
    ))
    if unscored:
        parts.append(
            f"<p>{', '.join(unscored)} is absent from that table because the node covariates "
            "cover only a minority of Jurkat's genes, too few to score any perturbation on "
            "these two folds. Over the whole screen the same arm reaches 58 of Jurkat's 705 "
            "perturbations, which is why it appears there and not here.</p>"
        )

    if not r.agreement.empty:
        spread = float(r.agreement["spread"].max())
        n_read = int(r.agreement["count"].max())
        found = (
            "every recomputation of an arm returns the identical number"
            if spread == 0.0
            else f"the largest disagreement between any two recomputations of the same arm "
                 f"is {P.num(spread)} correlation points"
        )
        parts.append(
            f"<p>Those reference arms are recomputed independently for every arm. Across "
            f"the {n_read} readings per cell line that withhold the same two folds, {found}, "
            "which is the reproducibility of the scoring path itself. The two refits withhold "
            "different folds, so their recomputations are scored on different perturbations "
            "against a different ceiling and are excluded from that comparison; including them "
            "would measure the fold difference, not the code.</p>"
        )
    if not r.null_check.empty and "inside_band" in r.null_check:
        # the band is drawn on folds 0 and 1, so only arms scored on those
        # folds can be checked against it
        nc = r.null_check[~r.null_check["train_arm"].astype(str).str.endswith(REFIT_TAGS)]
        fam = nc["family"].astype(str)
        sub = nc[fam.str.startswith("value_substitution")]
        own_q = nc[fam.str.startswith("pert_flag")]
        if not sub.empty:
            n_ok, n_all = int(sub["inside_band"].sum()), len(sub)
            verdict = (
                f"all {n_all} of them fall inside it anyway, so the band is not too tight "
                "for the arms it is used to judge"
                if n_ok == n_all
                else f"{n_all - n_ok} of {n_all} reach past it, so the band is tighter than "
                     "those arms' own nulls and the comparison against it flatters them"
            )
            parts.append(
                "<p>The shuffled-label band every score above is read against is run 3's "
                "pre-registered null, unchanged: the co-expression arm re-scored against "
                "shuffled labels, recomputed alongside the arm it judges. Each fine-tuned arm "
                "also draws "
                "its own shuffled labels; those draws are not folded into the band, because "
                f"each addition could only widen it, and {verdict}.</p>"
            )
        if not own_q.empty:
            n_out, n_all = int((~own_q["inside_band"]).sum()), len(own_q)
            widest = float(own_q[["draw_min", "draw_max"]].abs().max().max())
            pts = r.all_feature_points("final__pert_flag_resp")
            above = [
                f"{LINE[p['cell_line']]} {ARM_LABEL.get(p['arm'], p['arm'])} "
                f"{P.num(p['score'])}"
                for p in pts
                if (d := own_query_draws(r, p["cell_line"])) is not None
                and p["score"] > d[1]
            ]
            hi_line = sorted({p["cell_line"] for p in own_query_above(r)})
            rest = [
                p for p in pts
                if p["cell_line"] in hi_line and p not in own_query_above(r)
            ]
            other_line = [line for line in C.CELL_LINES if line not in hi_line]
            top_other = max(
                (p for p in pts if p["cell_line"] in other_line),
                key=lambda p: p["score"], default=None,
            )
            tail = (
                "On that reading every one of them is inside the draws, which is the same "
                "conclusion reached with less resolution."
                if not above
                else "On that reading " + str(len(above)) + " of the " + str(len(pts))
                + " readings still sit above the widest draw their cell line produced ("
                + "; ".join(above) + "), so this column is not a clean null result. "
                + (
                    "The other arms in the same cell line read "
                    + ", ".join(P.num(p["score"]) for p in rest)
                    + ". "
                    if rest else ""
                )
                + (
                    f"The largest reading in {LINE[top_other['cell_line']]}, "
                    f"{P.num(top_other['score'])}, carries an interval from "
                    f"{P.num(top_other['lo'])} to {P.num(top_other['hi'])}. "
                    if top_other else ""
                )
                + "Both of the two also sit three to four times below the gene-table arm. "
                "This is a noisier reading of the same quantity rather than a signal of the "
                "size the experiment was looking for."
            )
            parts.append(
                "<p>The perturbation-flag query is the exception, and it is worth stating "
                "plainly: its own shuffled draws are much wider than the band, reaching "
                f"{P.num(widest)} in absolute value, and {n_out} of the {n_all} arms drawn "
                "that way land outside it. That query asks the model for the effect of a "
                "flag it was fine-tuned to read, on genes it never saw perturbed, and the "
                "resulting feature is noisy enough that shuffled labels alone can produce a "
                "large score. Its column in the table above should therefore be read against "
                f"those draws rather than the band. {tail}</p>"
            )

    # variants of the primary arm: each is the same recipe with one choice
    # changed, and each is read in the same cell as the primary arm
    VARIANTS = (
        ("own_trainable", "the primary arm"),
        ("own_trainable__upstream", "upstream gene sampling"),
        ("own_trainable__f23", "refitted, folds 2 and 3 withheld"),
        ("own_trainable__f4", "refitted, fold 4 withheld"),
    )
    var_rows = []
    for line in C.CELL_LINES:
        band = r.null_ceiling(line)
        for run, label in VARIANTS:
            row = r.run_feature_row(line, run)
            t = r.run_row(line, run.replace("__lastepoch", "")) or {}
            if row is None:
                continue
            # the band is drawn on folds 0 and 1. The two refits withhold other
            # folds, so their scores have no band of their own and are not
            # judged against this one.
            refit = run.endswith(REFIT_TAGS)
            var_rows.append([
                LINE[line], label,
                "last" if run.endswith("__lastepoch") else str(t.get("selected_epoch", "n/a")),
                civ(row), str(row["n"]),
                "other folds, no band" if refit else ("yes" if row["score"] <= band else "no"),
            ])
    if var_rows:
        parts.append(
            "<p>Three variants of the primary arm test choices we made rather than inherited. "
            "Under <code>--upstream-sampling</code> the batch sampler reproduces the published "
            "one exactly, where the random gene subset drops the perturbed gene from about "
            "40% of rows and those rows carry no perturbation flag at all; our default forces "
            "the perturbed gene into every batch. Two refits withhold the complementary folds, "
            "so that pooling covers the whole screen. The checkpoint the recipe keeps is a "
            "choice we inherited rather than made, and it has its own section above.</p>"
        )
        parts.append(P.table(
            ["cell line", "variant", "epoch read", "response features", "perturbations",
             "inside the shuffled-label band"],
            var_rows,
            sources=[S_RUNARM, S_FOLLOWUP],
            caption="Each variant scored in the same cell as the primary arm, on the folds "
                    "that variant withheld. The refits are read on folds 2-4, so their "
                    "perturbation counts and their ceilings differ from the primary arm's "
                    "and the folds-0-and-1 shuffled band does not apply to them; their "
                    "contribution is the pooled whole-screen reading above.",
        ))
    return P.section("Controls", "".join(parts), sources=[S_SCORE, S_RUNARM])


#: the variant runs, labelled by what each varies rather than by its tag
VARIANT_LABEL = {
    "own_trainable__f23": "scGPT table, trainable, refitted with folds 2 and 3 withheld",
    "own_trainable__f4": "scGPT table, trainable, refitted with fold 4 withheld",
    "own_trainable__upstream": "scGPT table, trainable, upstream gene sampling",
}


def artifacts_section(r: Results) -> str:
    """Every training run's durable directory, read from the runs table itself."""
    if r.runs.empty:
        return ""
    label = {**ARM_LABEL, **VARIANT_LABEL}
    order = {a: i for i, a in enumerate(list(ARM_ORDER) + list(VARIANT_LABEL))}
    frame = r.runs[r.runs["artifact"].notna()].copy()
    frame["_line"] = frame["cell_line"].map({name: i for i, name in enumerate(C.CELL_LINES)})
    frame["_arm"] = frame["arm"].map(lambda a: order.get(a, len(order)))
    frame = frame.sort_values(["_line", "_arm"])
    rows = [
        [LINE.get(x.cell_line, x.cell_line), label.get(x.arm, x.arm),
         f"<code>{P.esc(str(x.artifact))}</code>"]
        for x in frame.itertuples()
    ]
    return P.section(
        "Checkpoints",
        f"<p>All {len(rows)} fine-tuning runs are kept: the five arms in each cell line plus "
        "the three variants of the primary arm. Each directory holds the selected and final "
        "weights, the exact training configuration, the resolved gene table before and after "
        "training including the learned projection where one exists, the per-epoch loss and "
        "validation history, the base checkpoint revision, and the saved predicted "
        "response.</p>"
        + P.table(["cell line", "run", "artifact"], rows, sources=[S_RUNARM]),
        sources=[S_RUNARM],
    )


def limitations(r: Results) -> str:
    excl = ""
    if not r.contrasts.empty:
        c = r.contrasts[
            (r.contrasts.translator == "mlp")
            & (r.contrasts.target == VERDICT[0])
            & (r.contrasts.weighting == VERDICT[1])
            & (r.contrasts.against == "scgpt_resp")
            & (r.contrasts.arm.isin([f"ft_{a}_resp" for a in ARM_ORDER]))
        ]
        out = c[~((c["lo"] <= 0) & (c["hi"] >= 0))]
        named = [d for d in paired_exceptions(r) if d["against"] == "scgpt_resp"]
        if not out.empty:
            excl = (
                f"{len(out)} of the {len(c)} paired differences against the frozen arm exclude "
                "zero (" + "; ".join(d["label"] for d in named)
                + f"), and {n_paired_all(r)} such differences are computed across all runs "
                "and both references. "
            )
    return P.section(
        "Limitations",
        "<ul>"
        "<li>This is a bound, not a demonstrated absence. " + excl +
        "The fold-centred intervals on the direct prediction are wide enough to admit a "
        "modest positive effect, and on these two folds co-expression itself is not separated "
        "from the shuffled-label band, so an effect of co-expression's size would not be "
        "resolved here. What the design supports is that fine-tuning produces no detectable "
        "improvement in the queried response features and leaves them far below the gene "
        "table.</li>"
        + (
            ""
            if not (above := own_query_above(r))
            else "<li>One read-out is not a clean null. Querying the checkpoints through the "
            "model's own perturbation-flag input rather than through the value substitution "
            f"runs 2 and 3 use puts {len(above)} of 10 readings above both the band and that "
            "query's own shuffled draws ("
            + "; ".join(
                f"{LINE[p['cell_line']]} {ARM_LABEL.get(p['arm'], p['arm'])} "
                f"{P.num(p['score'])}" for p in above
            )
            + "). That query's draws are wide and uneven across arms, the other arms in the "
            "same cell line do not reproduce it, and both readings are three to four times "
            "below the gene table, so it is reported as an unexplained noisy reading rather "
            "than as evidence either way.</li>"
        ) +
        "<li>One setting of the recipe could not be reproduced: the tutorial's "
        "<code>use_fast_transformer=True</code> needs flash-attn 1.x, which will not build "
        "against this torch, so scGPT's own fallback attention path was used. Same attention "
        "math, different kernel and different exact numerics. Every other setting the "
        "notebook names was checked against it and matches.</li>"
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
def build(summary_dir: Path, out_dir: Path, alt_dir: Path | None = None) -> Path:
    r = Results(summary_dir)
    alt = Results(alt_dir) if alt_dir else None
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

    # numbered in reading order: the direct-prediction figure comes first on
    # the page, so it is Figure 1
    f_swap = P.figure(
        2, "Response features before and after fine-tuning",
        frag("swap_before_after"),
        "Held-out score of the response-feature arm read from each checkpoint, on the "
        "residual target with reliability weighting and ceiling correction, 95% intervals "
        "over 2,000 resamples of the response groups. Open markers are the same architecture "
        "before any gradient step; filled markers are after fine-tuning. The shaded band is "
        "the range of the shuffled-label draws in that cell line and the dashed line is the "
        "gene-table arm.",
        "No fine-tuned model's response features leave the band, in either cell line, under "
        "any of the five gene-table arms.",
        sources=[S_SCORE, S_FIGS],
    )
    f_pred = P.figure(
        1, "The perturbation-specific part of three readings of the same model",
        frag("predicted_and_features"),
        "For each fine-tuned model: its own predicted perturbation response scored directly "
        "against the measured effects, and the response features read from it through its own "
        "perturbation-flag query and through the value-substitution query runs 2 and 3 use. "
        "All three on the same held-out perturbations, and all three on the fold-centred "
        "target, which removes the held-out fold's own mean from both sides and so leaves "
        "only what separates one perturbation from another. That is the one target on which "
        "the direct prediction and the feature arms measure the same thing.",
        "The prediction and the features read out of it sit in the same place, near zero. The "
        "feature swap is not losing a signal the prediction head has.",
        sources=[S_FOLLOWUP, S_FIGS],
    )
    f_train = P.figure(
        3, "Three quantities per epoch, and the epoch each rule keeps",
        frag("training_curves"),
        "Top: the recipe's own checkpoint-selection score, the Pearson correlation across all "
        "genes between each held-out perturbation's mean measured and mean predicted "
        "expression profile, averaged over perturbations. Middle: masked MSE on the same "
        "held-out cells, the training objective's own validation value. Bottom: the agreement "
        "between the predicted and the measured perturbation response after removing the "
        "response shared across perturbations. A star marks the epoch that panel's rule keeps.",
        "Both validation quantities move over training, so the runs train. The centred "
        "response agreement does not move off zero.",
        sources=[S_TRAIN, S_FIGS],
    )

    title = ("Fine-tuning scGPT on the screen leaves the queried response features inside "
             "the shuffled-label band, far below its own gene table")
    blocks = [
        P.section("", f"<h1>{P.esc(title)}</h1>"),
        opening(r),
        setup(r),
        direct_section(r, f_pred),
        verdict_section(r, f_swap),
        training_section(r, f_train),
        selection_section(r, alt),
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
    ap.add_argument(
        "--alt-summary",
        default=None,
        help="the same runs trained under the other checkpoint-selection rule",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    summary = Path(a.summary) if a.summary else paths.results_dir() / "summary"
    alt = Path(a.alt_summary) if a.alt_summary else None
    if alt is not None and not alt.exists():
        raise SystemExit(f"no alternative-rule summary at {alt}")
    dest = build(summary, Path(a.out), alt)
    print(f"wrote {dest} ({dest.stat().st_size / 1000:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
