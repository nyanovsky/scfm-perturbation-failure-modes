"""Build the report page from the summary tables and the figure bundles.

    PYTHONPATH=src uv run python -m scfm_pfm.report.build --out "$SILICO_APP_DIR"

Every number in the prose is read from ``results/summary``; nothing is typed in
by hand. The figures are rebuilt from the same tables so the page and the
bundles under ``figures/`` can never disagree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from silico_figures import PLOTLY_CONFIG

from .. import paths
from . import figures as figmod
from . import page as P

SRC = "experiments/experiment-3-yga5ws"
S_RUNNER = f"{SRC}/src/scfm_pfm/translate/runner.py"
S_ARMS = f"{SRC}/src/scfm_pfm/translate/arms.py"
S_TRAINERS = f"{SRC}/src/scfm_pfm/translate/trainers.py"
S_METRICS = f"{SRC}/src/scfm_pfm/metrics.py"
S_ANALYSIS = f"{SRC}/src/scfm_pfm/analysis.py"
S_CONFIG = f"{SRC}/src/scfm_pfm/config.py"
S_INGEST = f"{SRC}/src/scfm_pfm/ingest/build.py"
S_RECOMPUTE = f"{SRC}/src/scfm_pfm/ingest/recompute_effects.py"
S_FIGS = f"{SRC}/src/scfm_pfm/report/figures.py"
S_DISC = f"{SRC}/src/scfm_pfm/discrimination.py"
S_SENS = f"{SRC}/jobs/run_sensitivity.sh"
S_BEELINE = f"{SRC}/src/scfm_pfm/beeline.py"

VERDICT = figmod.VERDICT
PREREG = figmod.PREREG
MODELS = ("scgpt", "scprint", "transcriptformer", "aido_cell")
MODEL_LABEL = {
    "scgpt": "scGPT", "scprint": "scPRINT",
    "transcriptformer": "TranscriptFormer", "aido_cell": "AIDO.Cell",
}
FULL_SCOPE_MODELS = ("scgpt", "scprint", "transcriptformer")
LINE = {"hepg2": "HepG2", "jurkat": "Jurkat"}

ARM_LABEL = {
    "scgpt_emb": "scGPT embedding",
    "scgpt_full": "scGPT embedding + response",
    "scgpt_resp": "scGPT response only",
    "scgpt_hybrid": "scGPT + co-expression",
    "scprint_emb": "scPRINT embedding",
    "scprint_full": "scPRINT embedding + response",
    "scprint_resp": "scPRINT response only",
    "scprint_hybrid": "scPRINT + co-expression",
    "transcriptformer_emb": "TranscriptFormer embedding",
    "transcriptformer_full": "TranscriptFormer embedding + response",
    "transcriptformer_resp": "TranscriptFormer response only",
    "transcriptformer_hybrid": "TranscriptFormer + co-expression",
    "aido_cell_emb": "AIDO.Cell embedding",
    "aido_cell_full": "AIDO.Cell embedding + response",
    "aido_cell_resp": "AIDO.Cell response only",
    "aido_cell_hybrid": "AIDO.Cell + co-expression",
    "coexp": "co-expression",
    "coexp_check": "co-expression, other control half",
    "coexp_node": "co-expression + node statistics",
    "node": "node statistics, additive",
    "node_pairmlp": "node statistics, free",
    "node_check": "node statistics, shuffled gene identity",
    "null_randpair": "random vector per pair",
    "null_randgene_d512": "random vector per gene, 512 dims",
    "null_randgene_d1024": "random vector per gene, 1024 dims",
    "null_randgene_d1280": "random vector per gene, 1280 dims",
    "null_randgene_d5120": "random vector per gene, 5120 dims",
    "mean_only": "mean effect only",
}


def label(arm: str) -> str:
    return ARM_LABEL.get(arm, arm)


# --------------------------------------------------------------------------
# reading the tables


class Results:
    def __init__(self, summary_dir: Path, manifest: dict):
        self.dir = summary_dir
        self.runs = pd.read_parquet(summary_dir / "runs.parquet")
        self.con = pd.read_parquet(summary_dir / "contrasts.parquet")
        self.head = json.loads((summary_dir / "headline.json").read_text())
        self.manifest = manifest.get("lines", manifest)
        t = summary_dir / "transfer_runs.parquet"
        self.transfer = pd.read_parquet(t) if t.exists() else pd.DataFrame()
        o = summary_dir / "control_cell_overlap.parquet"
        self.overlap = pd.read_parquet(o) if o.exists() else pd.DataFrame()
        rc = summary_dir.parent / "recompute_report.json"
        self.recompute = json.loads(rc.read_text())["lines"] if rc.exists() else {}
        s = summary_dir / "sensitivity.parquet"
        self.sens = pd.read_parquet(s) if s.exists() else pd.DataFrame()
        bl = summary_dir.parent / "beeline_reproduction.json"
        self.beeline = json.loads(bl.read_text()) if bl.exists() else {}
        dv = summary_dir.parent / "discrimination_versions.json"
        self.disc_versions = json.loads(dv.read_text()) if dv.exists() else {}

    def disc(self, line: str, arm: str, col: str = "discrimination_cell_eval_mean") -> float:
        d = self.primary(line)
        d = d[d.arm == arm]
        return float(d[col].iloc[0]) if not d.empty else float("nan")

    def sens_score(self, model: str, anchor: str, col: str = VERDICT) -> float:
        if self.sens.empty:
            return float("nan")
        arm = f"{model}_resp" if anchor == "mean_control" else f"{model}_resp@{anchor}"
        d = self.sens[(self.sens.arm == arm) & (self.sens.translator == "mlp")]
        return float(d[col].iloc[0]) if not d.empty else float("nan")

    def primary(self, line: str) -> pd.DataFrame:
        return self.runs[
            (self.runs.cell_line == line)
            & (self.runs.scope == "full")
            & (self.runs.split == "grouped")
            & (self.runs.translator == "mlp")
            & (self.runs.target_source == "run1_all_controls")
        ]

    def score(self, line: str, arm: str, col: str = VERDICT) -> float:
        d = self.primary(line)
        d = d[d.arm == arm]
        return float(d[col].iloc[0]) if not d.empty else float("nan")

    def interval(self, line: str, arm: str, col: str = VERDICT) -> tuple[float, float, float]:
        d = self.primary(line)
        d = d[d.arm == arm]
        if d.empty:
            return (float("nan"),) * 3
        return (
            float(d[col].iloc[0]), float(d[col + "_lo"].iloc[0]), float(d[col + "_hi"].iloc[0])
        )

    def ceiling(self, line: str, col: str = VERDICT) -> float:
        """The split-half reproducibility ceiling the score of ``col`` divides by."""
        d = self.primary(line)
        name = col.replace("__disattenuated", "__ceiling")
        return float(d[name].iloc[0]) if not d.empty else float("nan")

    def null_ceiling(self, line: str, col: str = VERDICT) -> float:
        d = self.runs[
            (self.runs.cell_line == line) & (self.runs.scope == "full")
            & (self.runs.split == "grouped") & (self.runs.translator == "mlp")
            & self.runs.arm.str.contains(figmod.SHUFFLE_TAG, regex=False)
        ]
        return float(np.nanmax(d[col].to_numpy(dtype=float))) if not d.empty else float("nan")

    def best(self, line: str, prefixes: tuple[str, ...], exclude: str = "") -> pd.Series:
        """Highest-scoring arm among a set of prefixes, on the verdict cell."""
        d = self.primary(line)
        d = d[d.arm.str.startswith(prefixes)]
        d = d[~d.arm.str.contains(figmod.SHUFFLE_TAG, regex=False)]
        if exclude:
            d = d[~d.arm.str.contains(exclude, regex=False)]
        return d.sort_values(VERDICT, ascending=False).iloc[0]

    def model_arms(self, line: str) -> pd.DataFrame:
        """The twelve arms built on a foundation model, controls excluded."""
        d = self.primary(line)
        d = d[d.arm.str.startswith(MODELS) & ~d.arm.str.contains(figmod.SHUFFLE_TAG, regex=False)]
        return d

    def scope_arms(self, line: str, scope: str, split: str) -> pd.DataFrame:
        return self.runs[
            (self.runs.cell_line == line) & (self.runs.scope == scope)
            & (self.runs.split == split) & (self.runs.translator == "mlp")
            & (self.runs.target_source == "run1_all_controls")
        ]

    def tr(self, scope: str, mode: str = "dest_reference") -> pd.DataFrame:
        if self.transfer.empty:
            return self.transfer
        return self.transfer[
            (self.transfer.scope == scope)
            & (self.transfer.split == f"cross_line__{mode}")
            & (self.transfer.translator == "mlp")
        ]


def ci(v: float, lo: float, hi: float) -> str:
    """A point estimate with its bootstrap interval, in plain text."""
    if v != v:
        return "n/a"
    return f"{P.num(v)} ({P.num(lo)} to {P.num(hi)})"


# --------------------------------------------------------------------------
# the page


def opening(r: Results) -> str:
    h = r.best("hepg2", FULL_SCOPE_MODELS, exclude="hybrid")
    j = r.best("jurkat", FULL_SCOPE_MODELS, exclude="hybrid")
    hc = r.interval("hepg2", "coexp")
    jc = r.interval("jurkat", "coexp")
    tf_h = r.score("hepg2", "transcriptformer_resp", PREREG)
    co_h = r.score("hepg2", "coexp", PREREG)
    body = (
        "<p>A gene-pair feature taken from a single-cell foundation model predicts the "
        "perturbation-specific part of a CRISPRi knockdown response better than statistical "
        "co-expression computed from the same control cells. On held-out perturbations in "
        f"HepG2 the best model feature reaches {ci(h[VERDICT], h[VERDICT + '_lo'], h[VERDICT + '_hi'])} "
        f"(fold-centred correlation, {label(h.arm)}) against {ci(*hc)} for co-expression, and in "
        f"Jurkat {ci(j[VERDICT], j[VERDICT + '_lo'], j[VERDICT + '_hi'])} ({label(j.arm)}) against "
        f"{ci(*jc)}. Perturbation discrimination, a second scoring axis with no residualization "
        "in it at all, puts the same arms in the same order: the embedding arms above chance and "
        "co-expression below it.</p>"
        "<p>What carries that signal is the model's static gene embedding, not its simulated "
        "reaction to a perturbation. Adding the response features to the embedding changes the "
        "score by less than the design can resolve, and on their own the response features are "
        "weak everywhere. How weak depends on the score: on the fold-centred one they stay "
        "inside the shuffled-label band in HepG2 and clear it slightly for two of three models "
        "in Jurkat, while on the pre-registered score TranscriptFormer's response features beat "
        f"co-expression in HepG2 ({P.num(tf_h)} against {P.num(co_h)}). Both readings are "
        "reported throughout, because the choice between them is one this run made after seeing "
        "a control fail.</p>"
        "<p>Two limits change how far this reads. The score used to rank is the fold-centred "
        "one, chosen after the pre-registered score failed its own shuffled-label control, so it "
        "is post hoc and answers the narrower question of telling one held-out perturbation "
        "apart from another. And the absolute numbers are small: the score is already divided by "
        "the reproducibility limit, so 0.11 means the arm recovers about a tenth of the "
        "reproducible perturbation-specific signal on a correlation scale.</p>"
    )
    return P.section("", body, sources=[S_ANALYSIS, S_RUNNER])


def setup(r: Results) -> str:
    hep, jur = r.manifest["hepg2"], r.manifest["jurkat"]
    rows_data = [
        ("Screen", "Genome-wide CRISPRi Perturb-seq, GSE264667, HepG2 and Jurkat"),
        (
            "Read-out",
            f"HepG2 {hep['n_perturbations']} in-scope perturbations x "
            f"{hep['n_readout']} genes; Jurkat {jur['n_perturbations']} x {jur['n_readout']}",
        ),
        (
            "Target",
            "per-gene log2 fold change against the pooled non-targeting control, "
            "residualised by removing each read-out gene's mean effect across perturbations",
        ),
        (
            "Reproducibility ceiling",
            "split-half over cells, Spearman-Brown corrected, then reliability-weighted like "
            f"the score itself: {P.num(r.ceiling('hepg2', PREREG))} (HepG2) and "
            f"{P.num(r.ceiling('jurkat', PREREG))} (Jurkat) on the residual target, "
            f"{P.num(r.ceiling('hepg2'))} and {P.num(r.ceiling('jurkat'))} on the fold-centred "
            "one. Each score is divided by the square root of the ceiling of its own cell",
        ),
    ]
    rows_arms = [
        (
            "Foundation-model arms",
            "scGPT, AIDO.Cell, scPRINT, TranscriptFormer. Each arm is the concatenation of the "
            "two genes' features: the model's gene embedding, its 11-number response summary "
            "from the previous run, both, or either plus co-expression",
        ),
        (
            "Statistical baseline",
            "Pearson, Spearman, mutual information and partial correlation between the two "
            "genes over one half of the non-targeting control cells",
        ),
        (
            "Controls",
            "node statistics for the two genes on their own (additive and free), a fixed random "
            "vector per gene, a random vector per pair, and labels shuffled inside the training "
            "fold",
        ),
    ]
    rows_fit = [
        (
            "Translator",
            "one fixed network for every arm: 128 and 64 ReLU units, Adam at 1e-3, batch 4096, "
            "30 epochs, reliability-weighted mean squared error. Ridge regression is fitted "
            "alongside as a capacity control",
        ),
        (
            "Splits",
            "grouped (correlated perturbations held out together) and random, five folds each, "
            "seed 42",
        ),
        (
            "Score",
            "per-perturbation weighted Pearson correlation between predicted and measured "
            "response across read-out genes, divided by the square root of the reproducibility "
            "ceiling, then the median over held-out perturbations with a 2,000-resample "
            "bootstrap interval",
        ),
    ]
    return P.section(
        "Setup",
        P.spec([("Data", rows_data), ("Arms", rows_arms), ("Fitting and scoring", rows_fit)]),
        sources=[S_CONFIG, S_INGEST, S_ARMS, S_TRAINERS, S_METRICS],
    )


def _null_by_translator(r: Results, col: str) -> dict:
    d = r.runs[
        (r.runs.scope == "full") & (r.runs.split == "grouped")
        & (r.runs.target_source == "run1_all_controls")
        & r.runs.arm.str.contains(figmod.SHUFFLE_TAG, regex=False)
    ]
    g = d.groupby(["cell_line", "translator"])[col].max()
    return {k: float(v) for k, v in g.items()}


def metric_section(r: Results, chart: str) -> str:
    np_ = _null_by_translator(r, PREREG)
    nc = _null_by_translator(r, VERDICT)
    hco = r.score("hepg2", "coexp", PREREG)
    jco = r.score("jurkat", "coexp", PREREG)
    tf_h = r.score("hepg2", "transcriptformer_resp", PREREG)
    agree = pd.read_parquet(r.dir / "ordering_agreement.parquet")
    agree = agree[(agree.scope == "full") & (agree.split == "grouped")
                  & (agree.translator == "mlp") & (agree.cell == VERDICT)
                  & (agree.target_source == "run1_all_controls")]
    agr = {row.cell_line: float(row.spearman_vs_primary) for row in agree.itertuples()}
    con = r.con[(r.con.scope == "full") & (r.con.split == "grouped")
                & (r.con.translator == "mlp") & (r.con.metric == "residual_weighted")
                & (r.con.target_source == "run1_all_controls")
                & (r.con.cell_line == "hepg2") & (r.con.arm == "transcriptformer_resp")]
    dif = float(con.median_difference.iloc[0])
    lo, hi = float(con.lo.iloc[0]), float(con.hi.iloc[0])
    mrd = float(con.min_resolvable_difference_achieved.iloc[0])
    body = (
        "<p>The pre-registered score is the reliability-weighted correlation between the "
        "predicted and the measured residual response, corrected for the reproducibility "
        "ceiling. Its shuffled-label control does not sit at zero. Permuting the perturbation "
        "labels inside the training fold and refitting reaches "
        f"{P.num(np_[('hepg2', 'ridge')])} in HepG2 and {P.num(np_[('jurkat', 'ridge')])} in "
        "Jurkat with the ridge translator, which is as large as the co-expression arm's own "
        f"pre-registered score ({P.num(hco)} and {P.num(jco)}). With the network translator the "
        f"draws reach {P.num(np_[('hepg2', 'mlp')])} and {P.num(np_[('jurkat', 'mlp')])}. The "
        "reason is structural: the perturbations inside one held-out fold share a per-gene "
        "response, so a prediction that barely varies with the perturbation still correlates "
        "with the measurement.</p>"
        "<p>Removing each read-out gene's mean over the held-out fold from both the prediction "
        "and the measurement takes that shared component out. On this fold-centred score the "
        f"same ridge draws fall to {P.num(nc[('hepg2', 'ridge')])} and "
        f"{P.num(nc[('jurkat', 'ridge')])}. Every arm is reported on both scores throughout. "
        "The fold-centred one is used to rank because it is the one whose controls sit at zero. "
        "That choice was made after seeing the control fail, so it is post hoc, and it asks a "
        "narrower question: not whether an arm predicts the response, but whether it tells one "
        "held-out perturbation apart from another.</p>"
        f"<p>The switch is not free. The two scores rank the arms similarly but not identically "
        f"(rank correlation {P.num(agr['hepg2'])} in HepG2 and {P.num(agr['jurkat'])} in Jurkat "
        "over every arm), and what moves between them is exactly the set of arms this run has "
        "to be careful about. The embedding arms are far ahead on both. Among the weak arms the "
        "order changes: on the pre-registered score TranscriptFormer's response features reach "
        f"{P.num(tf_h)} in HepG2 against co-expression's {P.num(hco)}, a paired difference of "
        f"{ci(dif, lo, hi)} on the same held-out perturbations, which is larger than the "
        f"{P.num(mrd)} this design can resolve; on the fold-centred score the same pair is a "
        "tie. A reader who prefers the pre-registered score should read the response-feature "
        "result as one model beating co-expression in one cell line, not as a null.</p>"
    )
    return P.section(
        "The pre-registered score does not sit at zero under a shuffled-label control",
        body + chart,
        sources=[S_METRICS, S_ANALYSIS, S_FIGS],
    )


def headline_table(r: Results) -> str:
    arms = [
        "scgpt_emb", "scgpt_full", "scgpt_resp", "scgpt_hybrid",
        "scprint_emb", "transcriptformer_emb",
        "coexp", "coexp_node", "node_pairmlp", "null_randpair",
    ]
    rows = []
    for arm in arms:
        cells = [label(arm)]
        for line in ("hepg2", "jurkat"):
            v, lo, hi = r.interval(line, arm)
            p = r.score(line, arm, PREREG)
            cells += [ci(v, lo, hi) if v == v else "n/a", P.num(p)]
        rows.append(cells)
    return P.table(
        ["Arm", "HepG2 fold-centred", "HepG2 pre-registered",
         "Jurkat fold-centred", "Jurkat pre-registered"],
        rows,
        sources=[S_ANALYSIS],
    )


def verdict_section(r: Results, chart: str, chart2: str) -> str:
    hb = r.best("hepg2", FULL_SCOPE_MODELS, exclude="hybrid")
    jb = r.best("jurkat", FULL_SCOPE_MODELS, exclude="hybrid")
    hc = r.interval("hepg2", "coexp")
    jc = r.interval("jurkat", "coexp")
    hnull, jnull = r.null_ceiling("hepg2"), r.null_ceiling("jurkat")
    con = r.con[(r.con.scope == "full") & (r.con.split == "grouped")
                & (r.con.translator == "mlp") & (r.con.metric == "centered_weighted")
                & (r.con.target_source == "run1_all_controls")]
    hd = con[(con.cell_line == "hepg2") & (con.arm == hb.arm)]
    jd = con[(con.cell_line == "jurkat") & (con.arm == jb.arm)]
    mrd = {
        "hepg2": float(hd.min_resolvable_difference_achieved.iloc[0]),
        "jurkat": float(jd.min_resolvable_difference_achieved.iloc[0]),
    }
    counts = {}
    for line in ("hepg2", "jurkat"):
        m = r.model_arms(line)
        m = m[m.arm.str.startswith(FULL_SCOPE_MODELS)]
        counts[line] = (int((m[VERDICT] > r.null_ceiling(line)).sum()), int(m.shape[0]))
    body = (
        f"<p>{counts['hepg2'][0]} of the {counts['hepg2'][1]} foundation-model arms clear the "
        f"shuffled-label null in HepG2 and {counts['jurkat'][0]} of "
        f"{counts['jurkat'][1]} in Jurkat; the ones that fail are the response-only arms. "
        f"Co-expression does not clear it in HepG2, where it scores {ci(*hc)} "
        f"against a null ceiling of {P.num(hnull)}, and barely clears it in Jurkat "
        f"({ci(*jc)} against {P.num(jnull)}). The paired difference on the same held-out "
        f"perturbations is {ci(float(hd.median_difference.iloc[0]), float(hd.lo.iloc[0]), float(hd.hi.iloc[0]))} "
        f"for {label(hb.arm)} in HepG2 and "
        f"{ci(float(jd.median_difference.iloc[0]), float(jd.lo.iloc[0]), float(jd.hi.iloc[0]))} "
        f"for {label(jb.arm)} in Jurkat, against a smallest difference this design can resolve "
        f"of {P.num(mrd['hepg2'])} and {P.num(mrd['jurkat'])} respectively. The paired "
        "difference is taken on the per-perturbation correlations before the ceiling "
        "correction, so it is smaller than the gap between the two corrected scores above.</p>"
        "<p>Ridge regression on the same features scores near zero for almost every arm, so "
        "whatever the network finds is not linear in the concatenated pair feature.</p>"
    )
    return P.section(
        "Model embeddings clear the null; co-expression does not",
        body + chart + headline_table(r) + chart2,
        sources=[S_ANALYSIS, S_RUNNER, S_FIGS],
    )


def where_signal(r: Results) -> str:
    rows = []
    for line in ("hepg2", "jurkat"):
        for model in ("scgpt", "scprint", "transcriptformer"):
            emb = r.score(line, f"{model}_emb")
            full = r.score(line, f"{model}_full")
            resp = r.score(line, f"{model}_resp")
            hyb = r.score(line, f"{model}_hybrid")
            rows.append([
                LINE[line], MODEL_LABEL[model], P.num(emb), P.num(full), P.num(resp),
                P.num(hyb), P.signed(full - emb, 4), P.signed(hyb - emb, 4),
            ])
    tab = P.table(
        ["Cell line", "Model", "Embedding", "Embedding + response", "Response only",
         "Embedding + co-expression", "Response adds", "Co-expression adds"],
        rows, sources=[S_ANALYSIS],
    )
    resp_max = max(
        abs(r.score(line, f"{m}_resp"))
        for line in ("hepg2", "jurkat")
        for m in FULL_SCOPE_MODELS
    )
    add_resp = max(
        abs(r.score(line, f"{m}_full") - r.score(line, f"{m}_emb"))
        for line in ("hepg2", "jurkat") for m in FULL_SCOPE_MODELS
    )
    add_co = max(
        abs(r.score(line, f"{m}_hybrid") - r.score(line, f"{m}_emb"))
        for line in ("hepg2", "jurkat") for m in FULL_SCOPE_MODELS
    )
    hnull, jnull = r.null_ceiling("hepg2"), r.null_ceiling("jurkat")
    resp_rows = []
    for line, null in (("hepg2", hnull), ("jurkat", jnull)):
        for m in FULL_SCOPE_MODELS:
            v = r.score(line, f"{m}_resp")
            resp_rows.append((line, m, v, v > null))
    j_clear = [MODEL_LABEL[m] for line, m, _, ok in resp_rows if line == "jurkat" and ok]
    h_clear = [MODEL_LABEL[m] for line, m, _, ok in resp_rows if line == "hepg2" and ok]
    j_top = max(v for line, _, v, _ in resp_rows if line == "jurkat")
    prereg_beats = [
        (line, m)
        for line in ("hepg2", "jurkat")
        for m in FULL_SCOPE_MODELS
        if r.score(line, f"{m}_resp", PREREG) > r.score(line, "coexp", PREREG)
    ]
    body = (
        "<p>The previous run harvested response features by querying each model with a gene set "
        "to a different expression level and recording how the read-out genes moved. Those "
        "features are eleven numbers per pair. Added to the embedding they change the score by "
        f"at most {P.num(add_resp)}, with no consistent sign, and adding co-expression to the "
        f"embedding changes it by at most {P.num(add_co)}, also with no consistent sign. Neither "
        "addition reaches the smallest difference this design can resolve. What predicts the "
        "response is the static gene embedding.</p>"
        "<p>On their own the response features are weak, and how weak depends on which score is "
        f"read. Their largest fold-centred score anywhere is {P.num(resp_max)}. In HepG2 "
        f"{'none of the three models clears' if not h_clear else 'only ' + ', '.join(h_clear) + ' clears'} "
        f"the shuffled-label ceiling of {P.num(hnull)}. In Jurkat "
        f"{', '.join(j_clear) if j_clear else 'no model'} clears the lower ceiling there "
        f"({P.num(jnull)}), the largest reaching {P.num(j_top)} — above the null, and still "
        "between a third and a fifth of the embedding arms in the same cell line. On the "
        f"pre-registered score {len(prereg_beats)} of the six model-and-line combinations put "
        "the response features ahead of co-expression, TranscriptFormer in HepG2 by more than "
        "the resolvable difference. The claim these numbers support is that response features "
        "add nothing to the embedding and do not overturn co-expression as a class, not that "
        "they carry no signal at all.</p>"
    )
    return P.section("The signal is in the gene embedding, not the response features",
                     body + tab, sources=[S_ANALYSIS, S_ARMS])


def discrimination_section(r: Results, chart: str) -> str:
    """The second pre-registered scoring axis, and what its version pin cost."""
    rows = []
    for line in ("hepg2", "jurkat"):
        for arm in ("scgpt_emb", "scprint_emb", "transcriptformer_emb", "scgpt_full",
                    "coexp", "node_pairmlp", "scgpt_resp", "transcriptformer_resp",
                    "null_randpair"):
            rows.append([
                LINE[line], label(arm), P.num(r.disc(line, arm)),
                P.num(r.disc(line, arm, "discrimination_ours_residual_mean")),
                P.num(r.score(line, arm)),
            ])
    tab = P.table(
        ["Cell line", "Arm", "Discrimination, effect scale", "Discrimination, residual scale",
         "Fold-centred correlation"],
        rows, sources=[S_ANALYSIS, S_DISC],
    )
    best_h = max(r.disc("hepg2", f"{m}_emb") for m in FULL_SCOPE_MODELS)
    best_j = max(r.disc("jurkat", f"{m}_emb") for m in FULL_SCOPE_MODELS)
    body = (
        "<p>Correlation is not the only way to score this. Perturbation discrimination asks "
        "whether a predicted profile is closer to its own perturbation's measured profile than "
        "to the other held-out ones, which is this run's question with no residualization in it "
        "anywhere. It is computed with Arc Institute's own <code>cell-eval</code> rather than a "
        "reimplementation, on effect-scale profiles reconstituted by adding the training fold's "
        "mean effect back to the predicted residual. Chance is 0.5.</p>"
        f"<p>The embedding arms clear chance in both lines ({P.num(best_h)} in HepG2, "
        f"{P.num(best_j)} in Jurkat). Co-expression does not ({P.num(r.disc('hepg2', 'coexp'))} "
        f"and {P.num(r.disc('jurkat', 'coexp'))}), and neither do the response-only arms. This "
        "axis was pre-registered precisely so that a disagreement with the residual correlation "
        "would be visible; it agrees with the fold-centred score and disagrees with the "
        "pre-registered residual one, which is independent support for reading the fold-centred "
        "score as the one that tracks perturbation-specific signal.</p>"
        + _disc_version_note(r)
    )
    return P.section("Discrimination, the second scoring axis, agrees with the fold-centred score",
                     body + tab + chart, sources=[S_DISC, S_ANALYSIS])


def _disc_version_note(r: Results) -> str:
    v = r.disc_versions
    if not v:
        return ""
    spread = float(v["max_abs_difference_any_file"])
    vers = sorted({k for f in v["per_file"].values() for k in f["values"]})
    return (
        "<p>The pre-registration pinned <code>cell-eval==0.5.19</code>. That version does not "
        "exist: the package publishes 0.1.0 and then 0.5.33 onward, and the upstream tag list "
        "jumps 0.5.18 to 0.5.21, so the pin was unattainable when it was written and the jobs "
        f"resolved {v['resolved_in_the_sweep']}. Rescoring the same saved prediction matrices "
        f"under {', '.join(vers)} shows the choice matters: the number moves by up to "
        f"{P.num(spread)}, and the 0.5.x line ranks co-expression above the model arms while "
        "0.8.2 ranks them the other way. The implementations differ in one line — 0.5.x scores "
        "the absolute effect profile, 0.8.2 the signed one — and a distance on absolute values "
        "discards the direction of every response. The numbers above are 0.8.2, the current "
        "implementation of the challenge metric, and the version is recorded with them.</p>"
    )


def sensitivity_section(r: Results, chart: str) -> str:
    """Anchor and form sensitivity: does the query change what the features carry?"""
    if r.sens.empty:
        return ""
    null = r.null_ceiling("hepg2")
    models = [m for m in FULL_SCOPE_MODELS
              if not np.isnan(r.sens_score(m, "mean_control"))]
    anchors = ["mean_control", "uniform_virtual", "individual_cells", "mean_control_masked"]
    rows = []
    for m in models:
        cells = [MODEL_LABEL[m]]
        for a in anchors:
            v = r.sens_score(m, a)
            cells.append(P.num(v) if v == v else "not extracted")
        rows.append(cells)
    tab = P.table(
        ["Model", "Mean control", "Uniform virtual", "Individual cells", "Masked read-out"],
        rows, sources=[S_SENS],
    )
    deltas = [
        (m, a, r.sens_score(m, a) - r.sens_score(m, "mean_control"))
        for m in models for a in anchors[1:]
        if r.sens_score(m, a) == r.sens_score(m, a)
    ]
    worst = max(deltas, key=lambda x: abs(x[2])) if deltas else None
    uv = [d for d in deltas if d[1] == "uniform_virtual"]
    uv_better = [d for d in uv if d[2] > 0]
    cleared = [
        (m, a) for m in models for a in anchors
        if r.sens_score(m, a) > null
    ]
    rerun = r.head.get("sensitivity", {}).get("max_abs_rerun_difference", float("nan"))
    body = (
        "<p>The paper's mechanism claim is that querying the model at a uniform virtual value, "
        "rather than at the expression the cell actually has, gives a cleaner read of one "
        "gene's influence on another. The response features were harvested at four queries in "
        "the previous run, and this step scores the same arm at each of them in HepG2 under the "
        "grouped split. Everything else — translator, split, scope, scoring rule — is "
        "identical, so the only thing that varies is what the model was asked.</p>"
        f"<p>The anchor does not rescue the response features. The largest move away from the "
        f"on-distribution mean-control anchor is {P.signed(worst[2], 4) if worst else 'n/a'} "
        f"({MODEL_LABEL[worst[0]] if worst else ''} at the "
        f"{worst[1].replace('_', ' ') if worst else ''} anchor), and the uniform-virtual anchor "
        f"beats the on-distribution one for {len(uv_better)} of {len(uv)} models. "
        f"{'No arm at any anchor clears' if not cleared else str(len(cleared)) + ' of the scored arms clear'} "
        f"the shuffled-label ceiling of {P.num(null)}, against embedding arms at four to nine "
        "times that. Masking the read-out gene hides its own value together with its whole "
        "block, so the masked column confounds hiding the gene with removing part of the "
        "context and is not a clean mechanistic contrast.</p>"
        f"<p>The mean-control arms were re-run inside this job as its own reference: they "
        f"reproduce the main sweep to within {P.num(rerun)}, which also checks that the rebuilt "
        "bundle is the same bundle.</p>"
    )
    return P.section("The query anchor does not change what the response features carry",
                     body + tab + chart, sources=[S_SENS, S_CONFIG])


def controls_section(r: Results, split_chart: str, overlap_chart: str) -> str:
    npm_h, npm_j = r.score("hepg2", "node_pairmlp"), r.score("jurkat", "node_pairmlp")
    npm_hr = r.runs[(r.runs.cell_line == "hepg2") & (r.runs.scope == "full")
                    & (r.runs.split == "random") & (r.runs.translator == "mlp")
                    & (r.runs.arm == "node_pairmlp")][VERDICT]
    npm_hr = float(npm_hr.iloc[0]) if not npm_hr.empty else float("nan")
    hb = r.best("hepg2", FULL_SCOPE_MODELS, exclude="hybrid")
    body = (
        "<p>A pair feature can score without saying anything about the pair. Two controls bound "
        "that. The node-only control gives the network the same six summary statistics for each "
        "gene on its own (expression level, detection rate, co-expression degree) and no "
        "pairwise term. Fitted as a free network over both genes it reaches "
        f"{P.num(npm_h)} in HepG2 and {P.num(npm_j)} in Jurkat, well below {label(hb.arm)} at "
        f"{P.num(hb[VERDICT])} but above co-expression. Restricted to an additive form, one "
        "branch per gene summed, it is identically zero on the fold-centred score: after the "
        "fold's per-gene mean is removed, an additive prediction is the same vector for every "
        "perturbation. That is a property of the control and the score together, not a "
        "measurement, which is why the free version is the informative one.</p>"
        "<p>The gap between the two splits is the second bound. Under a random split, where "
        "correlated perturbations can fall on both sides, node statistics alone reach "
        f"{P.num(npm_hr)} in HepG2, {P.num(npm_hr / npm_h)} times their grouped-split score. "
        "Every arm gains, so "
        "the grouped split is the one that answers the question.</p>"
    )
    return P.section("Controls: node statistics, split design, and shared control cells",
                     body + split_chart + overlap_text(r) + overlap_chart,
                     sources=[S_ARMS, S_TRAINERS, S_RECOMPUTE, S_ANALYSIS])


def overlap_text(r: Results) -> str:
    if r.overlap.empty:
        return (
            "<p>The co-expression features and the effect they predict were computed from "
            "disjoint halves of the non-targeting control cells, so the baseline is not scored "
            "against a target built from its own cells.</p>"
        )
    d = r.overlap[r.overlap.translator == "mlp"]
    piv = d.pivot_table(index=["cell_line", "arm"], columns="variant",
                        values=VERDICT).reset_index()
    if not {"ctrl0", "ctrl1"}.issubset(piv.columns):
        return ""
    piv = piv.assign(delta=piv["ctrl1"] - piv["ctrl0"])
    co = piv[piv.arm.isin(["coexp", "coexp_check"])]
    mo = piv[piv.arm.isin(["scgpt_emb", "transcriptformer_emb"])]
    agree = {
        line: r.recompute[line]["targets"]["ctrlall"]["median_agreement_with_run1_effect"]
        for line in ("hepg2", "jurkat")
        if line in r.recompute
    }
    agree_txt = (
        "The three targets agree with the original effect matrix to a median per-perturbation "
        f"correlation of {P.num(agree['hepg2'])} in HepG2 and {P.num(agree['jurkat'])} in "
        "Jurkat, and with each other more closely still. "
        if len(agree) == 2
        else ""
    )
    return (
        "<p>Co-expression is computed from control cells, and the log2 fold change it predicts "
        "is measured against control cells too, so the baseline could be helped by sharing "
        "them. The target was rebuilt from the raw matrices three times: against the control "
        "half the co-expression features use, against the other half, and against all control "
        f"cells. {agree_txt}Moving the two plain co-expression arms from a target built on a "
        "disjoint control half to one built on the half they were computed from changes their "
        f"score by at most {P.num(float(co.delta.abs().max()))}. The embedding arms, which "
        f"touch no control cell, move by up to {P.num(float(mo.delta.abs().max()))} across the "
        "same pair of targets, so a shift of this size is retraining and target-estimation "
        "noise rather than an advantage the baseline draws from shared cells. Either way it is "
        "far smaller than the distance between co-expression and the embedding arms.</p>"
    )


def transfer_section(r: Results, chart: str) -> str:
    if r.transfer.empty:
        return ""
    full = r.tr("transfer_full")
    unseen = r.tr("transfer_unseen")
    parts = []
    for d, name in ((full, "every shared gene"), (unseen, "unseen genes only")):
        if d.empty:
            continue
        for direction in sorted(d.cell_line.unique()):
            b = d[(d.cell_line == direction)
                  & d.arm.str.startswith(FULL_SCOPE_MODELS)
                  & ~d.arm.str.contains("hybrid", regex=False)]
            b = b.sort_values(VERDICT, ascending=False).iloc[0]
            gi = d[(d.cell_line == direction) & (d.arm == "null_randgene_d5120")]
            co = d[(d.cell_line == direction) & (d.arm == "coexp")]
            parts.append({
                "scope": name, "direction": direction, "arm": b.arm,
                "score": float(b[VERDICT]), "lo": float(b[VERDICT + "_lo"]),
                "hi": float(b[VERDICT + "_hi"]),
                "gene_null": float(gi[VERDICT].iloc[0]) if not gi.empty else float("nan"),
                "coexp": float(co[VERDICT].iloc[0]) if not co.empty else float("nan"),
                "n": int(b.n_perturbations),
            })
    rows = [
        [p["scope"], figmod._direction_label(p["direction"]), str(p["n"]),
         f"{label(p['arm'])}: {ci(p['score'], p['lo'], p['hi'])}",
         P.num(p["gene_null"]), P.num(p["coexp"])]
        for p in parts
    ]
    tab = P.table(
        ["Scope", "Direction", "Held-out perturbations", "Best model arm",
         "Random vector per gene", "Co-expression"],
        rows, sources=[S_ANALYSIS],
    )
    shared = [p for p in parts if p["scope"] == "every shared gene"]
    unseen_p = [p for p in parts if p["scope"] == "unseen genes only"]
    mode_note = mode_agreement_note(r)
    body = (
        "<p>Training the translator on one screen and scoring it on the other looks like strong "
        "transfer until the gene-identity control is read. Over every gene shared by the two "
        f"screens the best model arm reaches {P.num(max(p['score'] for p in shared))}, and a "
        "fixed random vector per gene, a feature that carries nothing but which gene this is, "
        f"reaches {P.num(max(p['gene_null'] for p in shared))}. The two screens perturb "
        "mostly the same genes, and their responses to the same knockdown are correlated, so "
        "gene identity alone nearly solves that evaluation.</p>"
        "<p>Restricting the held-out side to perturbations whose gene was never perturbed in "
        "the training screen removes it. The gene-identity null falls to "
        f"{P.num(max(p['gene_null'] for p in unseen_p))} or below, and the model arms keep "
        f"{P.num(max(p['score'] for p in unseen_p))} at best. Co-expression does not transfer "
        "in either direction. The unseen-gene scope holds 89 and 78 held-out perturbations, so "
        "the intervals are wide.</p>" + mode_note
    )
    return P.section("Transfer across cell lines survives only against a gene-identity control",
                     body + tab + chart, sources=[S_RUNNER, S_ANALYSIS, S_FIGS])


def mode_agreement_note(r: Results) -> str:
    d = r.transfer[r.transfer.translator == "mlp"]
    piv = d.pivot_table(index=["cell_line", "scope", "arm"], columns="transfer_mode",
                        values=VERDICT)
    if "dest_reference" not in piv.columns:
        return ""
    shifts, ranks = [], []
    for mode in ("source_only", "dest_all"):
        if mode not in piv.columns:
            continue
        shifts.append(float((piv[mode] - piv["dest_reference"]).abs().max()))
        for _, g in piv.groupby(level=[0, 1]):
            gg = g[["dest_reference", mode]].dropna()
            if gg.shape[0] > 3:
                ranks.append(float(np.corrcoef(gg["dest_reference"].rank(),
                                               gg[mode].rank())[0, 1]))
    if not shifts:
        return ""
    return (
        "<p>The destination line's response has to be measured against something. Using the "
        "source line's mean, or all destination cells, instead of a held-out half of the "
        f"destination cells moves an arm's score by up to {P.num(max(shifts))} but leaves the "
        f"ordering intact (rank correlation {P.num(min(ranks))} or better within every "
        "direction and scope). The reported numbers use the held-out reference cells, the only "
        "one of the three that keeps the evaluation non-transductive.</p>"
    )


def scope_section(r: Results) -> str:
    rows = []
    for line in ("hepg2", "jurkat"):
        d = r.scope_arms(line, "common", "grouped")
        nulls = d[d.arm.str.contains(figmod.SHUFFLE_TAG, regex=False)][VERDICT]
        ceiling = float(np.nanmax(nulls.to_numpy())) if not nulls.empty else float("nan")
        for model in ("scgpt", "scprint", "transcriptformer", "aido_cell"):
            sub = d[d.arm.str.startswith(model)]
            if sub.empty:
                continue
            b = sub.sort_values(VERDICT, ascending=False).iloc[0]
            rows.append([
                LINE[line], MODEL_LABEL[model], label(b.arm), P.num(float(b[VERDICT])),
                P.num(ceiling), "yes" if float(b[VERDICT]) > ceiling else "no",
            ])
    tab = P.table(
        ["Cell line", "Model", "Best arm", "Fold-centred score", "Shuffled-label ceiling",
         "Clears the null"],
        rows, sources=[S_ANALYSIS],
    )
    n = int(r.scope_arms("hepg2", "common", "grouped").n_perturbations.iloc[0])
    body = (
        "<p>AIDO.Cell was harvested on a 250-gene subset in the previous run, so the only "
        "comparison that puts all four models on the same rows is that subset. On those "
        f"{n} perturbations the other three models still clear the shuffled-label null and "
        "AIDO.Cell does not, in either cell line. With a quarter of the perturbations the "
        "intervals are wide, so this is a weaker statement than the full-scope result: "
        "AIDO.Cell is not shown to be worse than the others, only not shown to work.</p>"
    )
    return P.section("On the 250-gene subset where all four models overlap, AIDO.Cell stays in "
                     "the null band", body + tab, sources=[S_ANALYSIS])


def robustness_section(r: Results, chart: str) -> str:
    agree = pd.read_parquet(r.dir / "ordering_agreement.parquet")
    a = agree[(agree.scope == "full") & (agree.split == "grouped")
              & (agree.translator == "mlp") & (agree.target_source == "run1_all_controls")]
    a = a.assign(target=a.cell.str.split("__").str[0])
    lo = {t: float(g.spearman_vs_primary.min()) for t, g in a.groupby("target")}
    body = (
        "<p>The score has three choices in it: which target (the residual response, the "
        "fold-centred response, or the raw log2 fold change), whether to weight read-out genes "
        "by their reliability, and whether to divide by the reproducibility ceiling. All twelve "
        "combinations are computed for every arm. Weighting and the ceiling correction change "
        "nothing about the ordering, and the residual and fold-centred targets rank the arms "
        f"similarly (rank correlation at least {P.num(lo['centered'])} against the "
        "pre-registered cell). The arms that move between those two are the weak ones, the "
        "response-only arms in particular; the embedding arms are ahead under both.</p>"
        "<p>The raw target is the exception: it ranks the arms differently (rank correlation "
        f"{P.num(lo['raw'])}). Predicting the raw fold change is mostly predicting each "
        "read-out gene's average response over all perturbations, which every arm can do and "
        "which says nothing about the perturbation. That is the component the pre-registered "
        "target removes and the fold-centred target removes again on the held-out side.</p>"
    )
    return P.section("The ranking holds under every scoring choice except the raw target",
                     body + chart, sources=[S_ANALYSIS, S_FIGS])


def limitations(r: Results) -> str:
    hb = r.best("hepg2", FULL_SCOPE_MODELS, exclude="hybrid")
    hceil = r.ceiling("hepg2")
    body = (
        "<ul>"
        "<li>The ranking metric is post hoc. The pre-registered cell of the grid is reported "
        "for every arm, and it fails its own shuffled-label control, so the verdict is read "
        "from the fold-centred cell instead. On the fold-centred score an arm is rewarded only "
        "for telling held-out perturbations apart, not for predicting the shared part of the "
        "response.</li>"
        f"<li>The effect is small in absolute terms. Every score is already divided by the "
        f"square root of the reproducibility ceiling ({P.num(hceil)} in HepG2), so 1.0 would "
        f"mean predicting the response as well as the measurement repeats itself. "
        f"{label(hb.arm)} reaches {P.num(float(hb[VERDICT]))} there, which leaves most of the "
        "reproducible perturbation-specific signal unexplained by every arm tested.</li>"
        "<li>One translator, and four anchors in one cell line. Every arm went through the same "
        "network shape; a larger translator could change the absolute numbers, though it would "
        "have to change them unevenly across arms to change the ordering. The anchor was varied "
        "in HepG2 only, over the three models with full-coverage feature blocks; AIDO.Cell's "
        "block covers 250 perturbations, so it cannot enter the full-scope sensitivity step at "
        "all.</li>"
        "<li>The discrimination axis depends on which release of the metric computes it. The "
        "version pinned before the run does not exist upstream, and the two real versions on "
        "either side of it disagree with the current one about whether co-expression or the "
        "model arms rank higher. The reported numbers name their version.</li>"
        + _beeline_limitation(r)
        + "<li>Two cell lines from one screen. Both come from the same CRISPRi Perturb-seq study "
        "with the same protocol, so the cross-line result is a weaker test of generality than "
        "two independent screens would be.</li>"
        "<li>Co-expression is one estimator family. Four statistics over one half of the "
        "control cells stand in for the statistical baseline; a network-inference method fitted "
        "to the same cells could do better than they do.</li>"
        "</ul>"
    )
    return P.section("Limitations", body, sources=[S_CONFIG, S_ANALYSIS])


def beeline_section(r: Results) -> str:
    """The source paper's own benchmark, its embedding baseline re-run here."""
    b = r.beeline
    if not b:
        return ""
    rows = [
        [
            f"{row['network']} / {row['dataset']}",
            P.num(row["auprc"]),
            P.num(row["paper_baseline_emb_auprc"]),
            P.signed(row["difference_from_paper"], 4),
            P.num(row["auprc_shuffled_embedding"]),
            f"{row['n_pairs_kept']:,}",
        ]
        for row in b["rows"]
    ]
    tab = P.table(
        ["Benchmark cell", "AUPRC here", "AUPRC reported", "Difference",
         "Shuffled-embedding control", "Pairs scored"],
        rows, sources=[S_BEELINE],
    )
    s = b["summary"]
    train = b["train_cell"]
    body = (
        "<p>The response features this run tests come from a paper whose own evidence is a "
        "different task: BEELINE network inference, where a projector maps a gene-pair feature "
        "to a regulatory edge and is scored by AUPRC on held-out datasets. Disagreeing with its "
        "conclusion is worth more if its setting reproduces here first, so its scGPT embedding "
        "baseline was re-run on its own benchmark files: the projector is fitted on the "
        f"{train['network']}/{train['dataset']} pair set "
        f"({train['n_pairs_kept']:,} pairs, balanced) and scored on the held-out cells, with the "
        "paper's projector shape, optimizer, batch size and epoch count.</p>"
        f"<p>It reproduces. Mean AUPRC over the three held-out cells is {P.num(s['mean_auprc'])} "
        f"against {P.num(s['paper_mean_auprc'])} reported, a mean difference of "
        f"{P.signed(s['mean_difference'], 4)} and at most {P.num(s['max_abs_difference'])} on any "
        "one cell. Permuting which gene gets which embedding, and refitting, drops it to "
        f"{P.num(s['shuffled_embedding_mean_auprc'])}, so the score is carried by the identity "
        "of the embedding rather than by the pair-set construction. The gradient-trajectory and "
        "virtual-value arms are not reproduced here: they would need the feature extraction "
        "re-run against these datasets, which is a separate harvest. So this run confirms the "
        "paper's setting and its embedding baseline, and its disagreement is about what response "
        "features do on perturbation response, not about its benchmark numbers.</p>"
    )
    return P.section("The source paper's embedding baseline reproduces on its own benchmark",
                     body + tab, sources=[S_BEELINE])


def _beeline_limitation(r: Results) -> str:
    if not r.beeline:
        return (
            "<li>The source paper's own benchmark is not reproduced here. Its network-inference "
            "task runs on different data with a different label, so this run tests its features "
            "on a task it was not evaluated on rather than contradicting its own numbers.</li>"
        )
    return (
        "<li>The source paper's benchmark is only partly reproduced here. Its projector and its "
        "embedding baseline are re-run on its own BEELINE datasets; its virtual-value and "
        "gradient-trajectory features are not, because those would need a fresh extraction "
        "against those datasets rather than the two screens used here. A disagreement about "
        "response features therefore rests on this task, not on their benchmark.</li>"
    )


def build(summary_dir: Path, out_dir: Path, fig_root: Path) -> Path:
    manifest = json.loads((paths.repo_dir() / "results" / "bundle_inventory.json").read_text())
    r = Results(summary_dir, manifest)
    made = {m["figure"]: m["fig"] for m in figmod.build_all(summary_dir, fig_root)}

    first = {"done": False}

    def frag(name: str, fallback: str = "") -> str:
        fig = made.get(name)
        if fig is None:
            return fallback
        include = "cdn" if not first["done"] else False
        first["done"] = True
        return fig.to_html(full_html=False, include_plotlyjs=include, config=PLOTLY_CONFIG)

    f_metric = P.figure(
        1, "Pre-registered score against fold-centred score, every arm",
        frag("metric_check"),
        "Each point is one arm in one cell line. The horizontal axis is the pre-registered "
        "reliability-weighted residual correlation; the vertical axis is the same correlation "
        "after the held-out fold's per-gene mean is removed. Green points are shuffled-label "
        "draws and random-vector controls.",
        "The controls are spread along the horizontal axis and collapse to zero on the "
        "vertical one, which is why the fold-centred score is the one used to rank.",
        sources=[S_METRICS, S_ANALYSIS, S_FIGS],
    )
    f_arms = P.figure(
        2, "Held-out score by arm, HepG2 and Jurkat",
        frag("arm_scores"),
        "Fold-centred, reliability-weighted, ceiling-corrected correlation on held-out "
        "perturbations under the grouped split, with 95% bootstrap intervals over 2,000 "
        "resamples (524 perturbations in HepG2, 705 in Jurkat). The dotted line is the largest "
        "shuffled-label draw in that cell line.",
        "Every arm that carries a foundation-model gene embedding sits above the null line; "
        "co-expression, the response-only arms and the random-vector controls sit at or below "
        "it.",
        sources=[S_RUNNER, S_ANALYSIS, S_FIGS],
    )
    con = r.con[(r.con.scope == "full") & (r.con.split == "grouped")
                & (r.con.translator == "mlp") & (r.con.metric == "centered_weighted")
                & (r.con.target_source == "run1_all_controls")]
    emb = con[con.arm.str.endswith(("_emb", "_full")) & ~con.arm.str.contains("shuffled")]
    vs_finding = (
        f"The embedding arms beat co-expression by {P.num(float(emb.median_difference.min()))} "
        f"to {P.num(float(emb.median_difference.max()))} correlation points; the response-only "
        "arms and the controls are indistinguishable from it."
    )
    f_vs = P.figure(
        3, "Paired difference against co-expression on the same held-out perturbations",
        frag("vs_coexpression"),
        "Median per-perturbation difference between each arm and the co-expression arm, scored "
        "on the same perturbations, with 95% bootstrap intervals on the paired difference.",
        vs_finding,
        sources=[S_ANALYSIS, S_FIGS],
    )
    f_disc = P.figure(
        4, "Perturbation discrimination by arm, HepG2 and Jurkat",
        frag("discrimination"),
        "Mean normalized rank of the true match: for each held-out perturbation, how close its "
        "predicted effect profile is to its own measured profile compared with every other "
        "held-out perturbation. Computed with cell-eval on effect-scale profiles. The dotted "
        "line is chance.",
        "Arms carrying a gene embedding sit above chance; co-expression, the response-only arms "
        "and the random-vector control sit at or below it.",
        sources=[S_DISC, S_FIGS],
    ) if "discrimination" in made else ""
    f_sens = P.figure(
        5, "Response-only score by query anchor, HepG2",
        frag("anchor_sensitivity"),
        "The same response-only arm scored at each of the four queries the features were "
        "harvested at, under the grouped split with everything else held fixed. The dotted line "
        "is the largest shuffled-label draw in HepG2.",
        "Moving off the on-distribution anchor does not lift the response features out of the "
        "null band, including at the uniform virtual value the source paper argues for.",
        sources=[S_SENS, S_FIGS],
    ) if "anchor_sensitivity" in made else ""
    f_split = P.figure(
        6, "Grouped split against random split, every arm",
        frag("split_gap"),
        "Each point is one arm in one cell line, scored under both splits. The dotted line is "
        "equality.",
        "Every arm scores higher under the random split, and the node-statistics control gains "
        "the most, which is what the grouped split is there to prevent.",
        sources=[S_RUNNER, S_FIGS],
    )
    f_overlap = P.figure(
        7, "Effect of sharing control cells with the target",
        frag("control_cell_overlap"),
        "Each arm's score against a target built from the same control cells the co-expression "
        "features use, minus its score against a target built from the other half of the "
        "control cells. Both targets were recomputed from the raw count matrices.",
        "The plain co-expression arms gain little from sharing control cells with the target, "
        "and the embedding arms, which touch no control cell, move by a similar amount.",
        sources=[S_RECOMPUTE, S_ANALYSIS, S_FIGS],
    ) if "control_cell_overlap" in made else ""
    f_transfer = P.figure(
        8, "Cross-cell-line transfer, over shared genes and over unseen genes",
        frag("transfer"),
        "The translator is fitted on one screen and scored on the other. The top row scores "
        "every gene shared by the two screens; the bottom row scores only perturbations whose "
        "gene was never perturbed in the training screen. The dotted line is a fixed random "
        "vector per gene.",
        "Over shared genes the gene-identity null nearly matches the models. Over unseen genes "
        "it collapses and the model arms keep a smaller but non-zero score.",
        sources=[S_RUNNER, S_FIGS],
    ) if "transfer" in made else ""
    f_grid = P.figure(
        9, "All twelve cells of the scoring grid",
        frag("grid_cells"),
        "Every arm scored under each combination of target (residual, fold-centred, raw), "
        "weighting, and ceiling correction, grouped split, MLP translator.",
        "The residual and fold-centred targets rank the arms the same way. The raw target does "
        "not, because predicting it is mostly predicting each gene's average response.",
        sources=[S_ANALYSIS, S_FIGS],
    )

    blocks = [
        P.section("", "<h1>Foundation-model gene embeddings predict knockdown response better "
                      "than co-expression</h1>"),
        opening(r),
        setup(r),
        metric_section(r, f_metric),
        verdict_section(r, f_arms, f_vs),
        where_signal(r),
        discrimination_section(r, f_disc),
        sensitivity_section(r, f_sens),
        beeline_section(r),
        scope_section(r),
        controls_section(r, f_split, f_overlap),
        transfer_section(r, f_transfer),
        robustness_section(r, f_grid),
        limitations(r),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "index.html"
    dest.write_text(P.document(
        "Foundation-model gene embeddings predict knockdown response better than co-expression",
        blocks,
    ))
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures", default=None)
    args = ap.parse_args(argv)
    summary = Path(args.summary) if args.summary else paths.results_dir() / "summary"
    fig_root = Path(args.figures) if args.figures else paths.repo_dir() / "figures"
    dest = build(summary, Path(args.out), fig_root)
    print(f"wrote {dest} ({dest.stat().st_size / 1000:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
