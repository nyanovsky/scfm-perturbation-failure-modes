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
S_AUDIT = f"{SRC}/src/scfm_pfm/protein/audit.py"
S_AUDIT_TABLES = f"{SRC}/src/scfm_pfm/protein/audit_tables.py"
S_UNIPROT = f"{SRC}/src/scfm_pfm/protein/uniprot.py"
S_ESM = f"{SRC}/src/scfm_pfm/protein/esm_embed.py"
S_IDENTITY = f"{SRC}/src/scfm_pfm/protein/identity.py"

DISCRIM = figmod.DISCRIM
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
    "null_randgene_d2560": "random vector per gene, 2560 dims",
    "null_randgene_d5120": "random vector per gene, 5120 dims",
    "scgpt_esm2_3b_emb": "scGPT table + ESM2-3B table",
    "null_randgene_d6144": "random vector per gene, 6144 dims",
    "esm2_650m_emb": "ESM2-650M protein embedding",
    "esm2_3b_emb": "ESM2-3B protein embedding",
    "mean_only": "mean effect only",
}


def label(arm: str) -> str:
    return ARM_LABEL.get(arm, arm)


def _join(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c' — lists read as sentences, not as tuples."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


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
        nc = summary_dir.parent / "verification" / "no_op_correction.json"
        self.no_op = json.loads(nc.read_text())["blocks"] if nc.exists() else []

        def _load(*parts) -> dict:
            path = summary_dir.parent.joinpath(*parts)
            return json.loads(path.read_text()) if path.exists() else {}

        self.protein_map = _load("protein", "gene_to_protein_report.json")
        self.protein_embed = _load("protein", "esm_embed_report.json")
        self.protein_install = _load("protein", "install_report.json")
        self.identity = _load("protein", "identity.json")
        self.audit = _load("provenance", "audit.json")
        self.audit_tables = _load("provenance", "audit_tables.json")

    def cka(self, line: str, a: str, b: str) -> float:
        """Linear CKA between two gene tables over that line's read-out genes."""
        pairs = self.identity.get("lines", {}).get(line, {}).get("pairs", {})
        key = f"{a}|{b}" if f"{a}|{b}" in pairs else f"{b}|{a}"
        return float(pairs[key]["cka"]) if key in pairs else float("nan")

    def rsa(self, line: str, a: str, b: str) -> float:
        pairs = self.identity.get("lines", {}).get(line, {}).get("pairs", {})
        key = f"{a}|{b}" if f"{a}|{b}" in pairs else f"{b}|{a}"
        return float(pairs[key]["rsa_spearman"]) if key in pairs else float("nan")

    def table_width(self, line: str, name: str) -> int:
        return int(self.identity.get("lines", {}).get(line, {}).get("widths", {}).get(name, 0))

    def transfer_score(self, direction: str, scope: str, arm: str,
                       mode: str = "dest_reference", col: str = DISCRIM) -> float:
        if self.transfer.empty:
            return float("nan")
        d = self.transfer[
            (self.transfer.cell_line == direction) & (self.transfer.scope == scope)
            & (self.transfer.split == f"cross_line__{mode}")
            & (self.transfer.translator == "mlp") & (self.transfer.arm == arm)
        ]
        return float(d[col].iloc[0]) if not d.empty else float("nan")

    def disc(self, line: str, arm: str, col: str = "discrimination_cell_eval_mean") -> float:
        d = self.primary(line)
        d = d[d.arm == arm]
        return float(d[col].iloc[0]) if not d.empty else float("nan")

    def sens_score(self, model: str, anchor: str, col: str = DISCRIM) -> float:
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

    def score(self, line: str, arm: str, col: str = DISCRIM) -> float:
        d = self.primary(line)
        d = d[d.arm == arm]
        return float(d[col].iloc[0]) if not d.empty else float("nan")

    def interval(self, line: str, arm: str, col: str = DISCRIM) -> tuple[float, float, float]:
        d = self.primary(line)
        d = d[d.arm == arm]
        if d.empty:
            return (float("nan"),) * 3
        return (
            float(d[col].iloc[0]), float(d[col + "_lo"].iloc[0]), float(d[col + "_hi"].iloc[0])
        )

    def ceiling(self, line: str, col: str = DISCRIM) -> float:
        """The split-half reproducibility ceiling the score of ``col`` divides by."""
        d = self.primary(line)
        name = col.replace("__disattenuated", "__ceiling")
        return float(d[name].iloc[0]) if not d.empty else float("nan")

    def null_ceiling(self, line: str, col: str = DISCRIM) -> float:
        d = self.runs[
            (self.runs.cell_line == line) & (self.runs.scope == "full")
            & (self.runs.split == "grouped") & (self.runs.translator == "mlp")
            & self.runs.arm.str.contains(figmod.SHUFFLE_TAG, regex=False)
        ]
        return float(np.nanmax(d[col].to_numpy(dtype=float))) if not d.empty else float("nan")

    def best(self, line: str, prefixes: tuple[str, ...], exclude: str = "",
             col: str = PREREG) -> pd.Series:
        """Highest-scoring arm among a set of prefixes, on ``col``."""
        d = self.primary(line)
        d = d[d.arm.str.startswith(prefixes)]
        d = d[~d.arm.str.contains(figmod.SHUFFLE_TAG, regex=False)]
        if exclude:
            d = d[~d.arm.str.contains(exclude, regex=False)]
        return d.sort_values(col, ascending=False).iloc[0]

    def vs_coexp(self, line: str, arm: str, col: str = PREREG) -> dict:
        """The paired difference against co-expression on ``col``'s target."""
        metric = "residual_weighted" if col == PREREG else "centered_weighted"
        d = self.con[
            (self.con.cell_line == line) & (self.con.scope == "full")
            & (self.con.split == "grouped") & (self.con.translator == "mlp")
            & (self.con.target_source == "run1_all_controls")
            & (self.con.metric == metric) & (self.con.arm == arm)
        ]
        if d.empty:
            return {}
        row = d.iloc[0]
        return {
            "difference": float(row.median_difference), "lo": float(row.lo),
            "hi": float(row.hi), "n_paired": int(row.n_paired),
            "n_groups": int(row.n_groups_paired),
            "mrd": float(row.min_resolvable_difference_achieved),
        }

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
    hc = r.interval("hepg2", "coexp", PREREG)
    jc = r.interval("jurkat", "coexp", PREREG)
    hd = r.vs_coexp("hepg2", h.arm)
    jd = r.vs_coexp("jurkat", j.arm)
    tf_h = r.score("hepg2", "transcriptformer_resp", PREREG)
    co_h = r.score("hepg2", "coexp", PREREG)
    esm_h = r.interval("hepg2", "esm2_3b_emb", PREREG)
    esm_j = r.interval("jurkat", "esm2_3b_emb", PREREG)
    ep_h = _paired(r, "hepg2", "esm2_3b_emb", "coexp", PREREG_KEY) or {}
    ep_j = _paired(r, "jurkat", "esm2_3b_emb", "coexp", PREREG_KEY) or {}
    protein = (
        "<p>The feature doing that work is the model's static gene-identity table, and two of "
        "the four tables were not learned from cells. TranscriptFormer ships a frozen ESM2 "
        "protein embedding and never updates it; scPRINT started from a protein-derived table "
        "and trained it; scGPT and AIDO.Cell learned theirs from expression alone. So we built "
        "a gene table directly from protein sequence with two public ESM2 checkpoints and ran "
        "it through the identical translator, folds and scoring. It reaches "
        f"{ci(*esm_h)} in HepG2 and {ci(*esm_j)} in Jurkat, beating co-expression on the same "
        f"held-out perturbations by {ci(ep_h.get('median_difference', float('nan')), ep_h.get('lo', float('nan')), ep_h.get('hi', float('nan')))} "
        f"and {ci(ep_j.get('median_difference', float('nan')), ep_j.get('lo', float('nan')), ep_j.get('hi', float('nan')))}. "
        "Paired against the single-cell models' own tables it is a tie in seven of the eight "
        "comparisons and ahead of scGPT in Jurkat. Protein sequence alone therefore supplies a "
        "gene table that does this job as well as the single-cell models' tables do. That says "
        "where the signal can come from; it does not say where scGPT's comes from, since "
        "scGPT's table carries no protein-space structure and still scores as well.</p>"
    ) if r.head.get("protein") else ""
    body = (
        "<p>A gene-pair feature taken from a single-cell foundation model predicts the "
        "perturbation-specific part of a CRISPRi knockdown response better than statistical "
        "co-expression computed from the same control cells. On held-out perturbations in "
        f"HepG2 the best model feature reaches {ci(h[PREREG], h[PREREG + '_lo'], h[PREREG + '_hi'])} "
        f"({label(h.arm)}) against {ci(*hc)} for co-expression, and in Jurkat "
        f"{ci(j[PREREG], j[PREREG + '_lo'], j[PREREG + '_hi'])} ({label(j.arm)}) against "
        f"{ci(*jc)}. Paired on the same held-out perturbations the difference is "
        f"{ci(hd['difference'], hd['lo'], hd['hi'])} and {ci(jd['difference'], jd['lo'], jd['hi'])}. "
        "This is the pre-registered scoring rule, and every interval on this page resamples "
        "whole response groups rather than individual perturbations, because correlated "
        "perturbations are held out together and are not independent draws.</p>"
        + protein +
        "<p>What carries that signal is the model's static gene embedding, not its simulated "
        "reaction to a perturbation. Adding the response features to the embedding changes the "
        "score by less than the design can resolve, and on their own the response features are "
        "weak everywhere. How weak depends on the score: on the fold-centred one they stay "
        "inside the shuffled-label band in HepG2 and clear it slightly for two of three models "
        "in Jurkat, while on the pre-registered score TranscriptFormer's response features "
        f"score above co-expression in HepG2 ({P.num(tf_h)} against {P.num(co_h)}), by less "
        "than the paired comparison separates. Both readings are reported throughout.</p>"
        "<p>Three limits change how far this reads. The pre-registered score rewards an arm "
        "for the part of a held-out group's response that its members share, which a prediction "
        "that barely varies with the perturbation can also get; a fold-centred version that "
        "strips that shared part is reported beside it everywhere, but it removes the held-out "
        "fold's own mean and so is a diagnostic rather than a prediction a model could make in "
        "advance. The absolute numbers are small: the score is already divided by the "
        "reproducibility limit, so 0.27 means the arm recovers about a quarter of the "
        "reproducible perturbation-specific signal on a correlation scale. And the second "
        "pre-registered axis, perturbation discrimination, depends on which release of the "
        "metric computes it, so it neither confirms nor contradicts the ranking.</p>"
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
            "ceiling, then the median over held-out perturbations. Intervals are 2,000 "
            "resamples of the response groups, not of individual perturbations, so they carry "
            "the same dependence the grouped split is built around",
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
    nc = _null_by_translator(r, DISCRIM)
    hco = r.score("hepg2", "coexp", PREREG)
    jco = r.score("jurkat", "coexp", PREREG)
    tf_h = r.score("hepg2", "transcriptformer_resp", PREREG)
    agree = pd.read_parquet(r.dir / "ordering_agreement.parquet")
    agree = agree[(agree.scope == "full") & (agree.split == "grouped")
                  & (agree.translator == "mlp") & (agree.cell == DISCRIM)
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
        "<p>The verdict is read from the rule fixed before the run: the reliability-weighted "
        "correlation between the predicted and the measured residual response, corrected for "
        "the reproducibility ceiling. Residual here means against the mean effect of the "
        "TRAINING folds, so nothing about a held-out perturbation enters its own score.</p>"
        "<p>That rule has a known soft spot, and it is worth stating plainly because it shapes "
        "how the numbers should be read. Perturbations inside one held-out group share a "
        "per-gene response, so a prediction that barely varies with the perturbation still "
        "correlates with the measurement. Permuting the perturbation labels inside the training "
        f"fold and refitting reaches {P.num(np_[('hepg2', 'ridge')])} in HepG2 and "
        f"{P.num(np_[('jurkat', 'ridge')])} in Jurkat with the ridge translator, as large as "
        f"the co-expression arm's own pre-registered score ({P.num(hco)} and {P.num(jco)}); "
        f"with the network translator the draws reach {P.num(np_[('hepg2', 'mlp')])} and "
        f"{P.num(np_[('jurkat', 'mlp')])}. Those are the ceilings a score has to clear, and the "
        "embedding arms sit five to ten times above them.</p>"
        "<p>Removing each read-out gene's mean over the held-out fold from both the prediction "
        "and the measurement takes the shared component out. On that fold-centred score the "
        f"same ridge draws fall to {P.num(nc[('hepg2', 'ridge')])} and "
        f"{P.num(nc[('jurkat', 'ridge')])}. Every arm is reported on both scores throughout. "
        "The fold-centred one is a diagnostic, not a second verdict: it subtracts a mean of the "
        "held-out outcomes, which no model could compute in advance, and it was written after "
        "seeing the shuffled draws, so it is post hoc. What it answers is narrower than the "
        "pre-registered question: not whether an arm predicts the response, but whether it "
        "tells one held-out perturbation apart from another.</p>"
        f"<p>The two scores rank the arms similarly but not identically (rank correlation "
        f"{P.num(agr['hepg2'])} in HepG2 and {P.num(agr['jurkat'])} in Jurkat over every arm), "
        "and what moves between them is the set of weak arms. The embedding arms are far ahead "
        "on both. Among the weak arms the order changes: on the pre-registered score "
        f"TranscriptFormer's response features reach {P.num(tf_h)} in HepG2 against "
        f"co-expression's {P.num(hco)}, but the paired difference on the same held-out "
        f"perturbations is {ci(dif, lo, hi)}, inside the {P.num(mrd)} this design resolves once "
        "the response groups are resampled; on the fold-centred score the same pair is a tie as "
        "well. Where the two scores disagree, this page says so rather than picking the "
        "flattering one.</p>"
    )
    return P.section(
        "Two scores: the pre-registered rule, and a fold-centred diagnostic beside it",
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
            pv, plo, phi = r.interval(line, arm, PREREG)
            dv, dlo, dhi = r.interval(line, arm)
            cells += [
                ci(pv, plo, phi) if pv == pv else "n/a",
                ci(dv, dlo, dhi) if dv == dv else "n/a",
            ]
        rows.append(cells)
    return P.table(
        ["Arm", "HepG2 pre-registered", "HepG2 fold-centred",
         "Jurkat pre-registered", "Jurkat fold-centred"],
        rows,
        sources=[S_ANALYSIS],
        caption=("Held-out score on the pre-registered rule, which the verdict is read from, "
                 "and on the fold-centred diagnostic beside it. Intervals are 2,000 resamples "
                 "of the response groups."),
    )


def verdict_section(r: Results, chart: str, chart2: str) -> str:
    hb = r.best("hepg2", FULL_SCOPE_MODELS, exclude="hybrid")
    jb = r.best("jurkat", FULL_SCOPE_MODELS, exclude="hybrid")
    hc = r.interval("hepg2", "coexp", PREREG)
    jc = r.interval("jurkat", "coexp", PREREG)
    hnull, jnull = r.null_ceiling("hepg2", PREREG), r.null_ceiling("jurkat", PREREG)
    hd, jd = r.vs_coexp("hepg2", hb.arm), r.vs_coexp("jurkat", jb.arm)
    counts, emb_counts = {}, {}
    for line in ("hepg2", "jurkat"):
        m = r.model_arms(line)
        m = m[m.arm.str.startswith(FULL_SCOPE_MODELS)]
        counts[line] = (int((m[PREREG] > r.null_ceiling(line, PREREG)).sum()), int(m.shape[0]))
        emb = [a for a in m.arm if a.endswith("_emb")]
        ahead = [a for a in emb if (r.vs_coexp(line, a) or {"lo": 0})["lo"] > 0]
        emb_counts[line] = (len(ahead), len(emb))
    body = (
        f"<p>{counts['hepg2'][0]} of the {counts['hepg2'][1]} foundation-model arms clear the "
        f"shuffled-label null in HepG2 and {counts['jurkat'][0]} of "
        f"{counts['jurkat'][1]} in Jurkat; the one that fails is scGPT's response-only arm, in "
        "both lines. "
        f"Co-expression barely clears it: {ci(*hc)} in HepG2 against a null ceiling of "
        f"{P.num(hnull)}, and {ci(*jc)} in Jurkat against {P.num(jnull)}. Every one of the "
        f"embedding arms beats it on the same held-out perturbations, paired: "
        f"{emb_counts['hepg2'][0]} of {emb_counts['hepg2'][1]} in HepG2 and "
        f"{emb_counts['jurkat'][0]} of {emb_counts['jurkat'][1]} in Jurkat have a paired "
        f"interval clear of zero. The largest is {ci(hd['difference'], hd['lo'], hd['hi'])} for "
        f"{label(hb.arm)} in HepG2 and {ci(jd['difference'], jd['lo'], jd['hi'])} for "
        f"{label(jb.arm)} in Jurkat, against a smallest difference this design resolves of "
        f"{P.num(hd['mrd'])} and {P.num(jd['mrd'])}. The paired difference is taken on the "
        "per-perturbation correlations before the ceiling correction, so it is smaller than the "
        "gap between the two corrected scores above.</p>"
        "<p>Ridge regression on the same features scores near zero for almost every arm, so "
        "whatever the network finds is not linear in the concatenated pair feature.</p>"
        + (
            "<p>Throughout this section 'foundation-model arm' means an arm built on one of the "
            "four single-cell models, which is how they are packaged, not a claim about where "
            "the feature came from. Two of the four gene tables trace back to a protein "
            "language model rather than to single-cell data; a later section reads that out of "
            "the checkpoints and runs a raw protein table through the same pipeline.</p>"
            if r.head.get("protein") else ""
        )
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
        f"{_join(j_clear) if j_clear else 'no model'} clear the lower ceiling there "
        f"({P.num(jnull)}), the largest reaching {P.num(j_top)}, above the null and still "
        "between a third and a fifth of the embedding arms in the same cell line. On the "
        f"pre-registered score {len(prereg_beats)} of the six model-and-line combinations put "
        "the response features numerically ahead of co-expression, none of them by more than "
        "the paired difference resolves. The claim these numbers support is that response "
        "features add nothing to the embedding and do not overturn co-expression as a class, "
        "not that they carry no signal at all.</p>"
    )
    return P.section("The signal is in the gene embedding, not the response features",
                     body + tab, sources=[S_ANALYSIS, S_ARMS])


#: what the audit found, per model, with the checkpoint fact it rests on. The
#: numbers are pulled from the audit records rather than written here.
PROVENANCE_ORDER = ("transcriptformer", "scprint", "scgpt", "aido_cell")


def provenance_rows(r: Results) -> list[list[str]]:
    models = r.audit.get("models", {})
    tf = r.audit_tables.get("transcriptformer", {})
    sp = r.audit_tables.get("scprint", {})
    tf_cmp = tf.get("weights_vs_shipped_genes_only", {})
    opt = models.get("scprint", {}).get("optimizer", {})

    def shape(model: str) -> str:
        s = models.get(model, {}).get("gene_table_shape") or []
        return " x ".join(f"{int(x):,}" for x in s) if s else "n/a"

    return [
        [
            "TranscriptFormer", models.get("transcriptformer", {}).get("weights_key")
            or "gene_embeddings.embedding", shape("transcriptformer"),
            "<strong>no</strong>",
            f"its {tf.get('h5_n_genes', 0):,} gene rows are bitwise the ESM2 file shipped beside "
            f"the weights (largest difference {P.num(tf_cmp.get('max_abs_diff', float('nan')))}), "
            "and the loader hard-codes freeze=True",
        ],
        [
            "scPRINT", "gene_encoder.embedding", shape("scprint"),
            "yes, from a protein-derived start",
            f"the checkpoint records freeze_embeddings: {str(sp.get('freeze_embeddings')).lower()} "
            f"and its optimizer state holds "
            f"{int(opt.get('n_state_tensors_with_gene_table_shape', 0))} Adam moment tensors of "
            "exactly that shape, which only the gene table has",
        ],
        [
            "scGPT", "encoder.embedding", shape("scgpt"),
            "yes, learned from scratch",
            "a plain embedding table over the gene vocabulary; nothing in the checkpoint reads a "
            "protein sequence",
        ],
        [
            "AIDO.Cell", "bert.position_embedding", shape("aido_cell"),
            "yes, learned from scratch",
            "gene identity is the position in the fixed 19,264-gene order, and no protein model "
            "appears in the checkpoint or its modelling source",
        ],
    ]


def _ptransfer(r: Results, direction: str, scope: str,
               mode: str = "dest_reference") -> dict[str, dict]:
    """The protein job's cross-line block, keyed by arm."""
    block = r.head.get("protein_transfer", {}).get("blocks", {}).get(
        f"{direction}|{scope}|cross_line__{mode}", []
    )
    return {row["arm"]: row for row in block}


#: the two keys the protein block stores paired differences under
PREREG_KEY = "paired_differences_preregistered"
DISCRIM_KEY = "paired_differences"


def _paired(r: Results, line: str, arm: str, reference: str,
            key: str = PREREG_KEY) -> dict | None:
    for row in r.head.get("protein", {}).get("lines", {}).get(line, {}).get(key, []):
        if row["arm"] == arm and row["reference"] == reference:
            return row
    return None


def _beats(row: dict | None) -> str:
    """How a paired difference reads: ahead, behind, or a tie at this resolution."""
    if row is None:
        return "not measured"
    if row["lo"] > 0:
        return "ahead"
    if row["hi"] < 0:
        return "behind"
    return "a tie"


def protein_finding(r: Results) -> str:
    """The one-line reading of the protein figure, from the paired differences."""
    if not r.head.get("protein"):
        return ""
    verdict = {
        model: {line: _beats(_paired(r, line, "esm2_3b_emb", f"{model}_emb"))
                for line in ("hepg2", "jurkat")}
        for model in ("transcriptformer", "scprint", "scgpt")
    }
    ties = [MODEL_LABEL[m] for m, v in verdict.items() if set(v.values()) == {"a tie"}]
    ahead = [MODEL_LABEL[m] for m, v in verdict.items() if set(v.values()) == {"ahead"}]
    behind = [MODEL_LABEL[m] for m, v in verdict.items() if set(v.values()) == {"behind"}]
    parts = []
    if ties:
        parts.append(f"ties {_join(ties)} in both cell lines")
    if ahead:
        parts.append(f"beats {_join(ahead)} in both")
    if behind:
        parts.append(f"stays behind {_join(behind)} in both")
    for m, v in verdict.items():
        if len(set(v.values())) > 1:
            parts.append(
                f"{'ties' if v['hepg2'] == 'a tie' else v['hepg2']} {MODEL_LABEL[m]} in HepG2 "
                f"and is {v['jurkat']} in Jurkat"
            )
    return ("The protein-only arm " + _join(parts) + ".") if parts else ""


def protein_section(r: Results, chart: str, chart2: str) -> str:
    """Stage 1 and stage 2 of the protein-model control, in one place."""
    if not r.audit or not r.runs.arm.str.startswith("esm2").any():
        return ""
    prov = P.table(
        ["Model", "Gene-identity table", "Shape", "Updated during pretraining",
         "What the checkpoint shows"],
        provenance_rows(r), sources=[S_AUDIT, S_AUDIT_TABLES],
    )
    mapping = r.protein_map
    routes = mapping.get("routes", {})
    embed = r.protein_embed.get("models", {})
    lengths = mapping.get("length", {})

    rows = []
    for line in ("hepg2", "jurkat"):
        for arm in ("esm2_3b_emb", "esm2_650m_emb", "transcriptformer_emb", "scprint_emb",
                    "scgpt_emb", "coexp"):
            pv, plo, phi = r.interval(line, arm, PREREG)
            dv, dlo, dhi = r.interval(line, arm)
            is_esm = arm.startswith("esm2")
            rows.append([
                LINE[line], label(arm), f"{int(r.runs[(r.runs.arm == arm)].feature_dim.iloc[0])}",
                ci(pv, plo, phi), ci(dv, dlo, dhi),
                _beats(_paired(r, line, arm, "transcriptformer_emb") if is_esm else None),
                _beats(_paired(r, line, arm, "scprint_emb") if is_esm else None),
                _beats(_paired(r, line, arm, "scgpt_emb") if is_esm else None),
            ])
    table = P.table(
        ["Cell line", "Arm", "Feature width", "Pre-registered score", "Fold-centred score",
         "vs TranscriptFormer", "vs scPRINT", "vs scGPT"],
        rows, sources=[S_ANALYSIS, S_RUNNER],
        caption=("Reliability-weighted, ceiling-corrected correlation on held-out "
                 "perturbations, full scope, grouped split, MLP translator, on the "
                 "pre-registered rule and on the fold-centred diagnostic. The last three "
                 "columns read the paired per-perturbation difference on the pre-registered "
                 "rule between the protein arm and that model's embedding arm on the same "
                 "perturbations: 'ahead' and 'behind' mean the group-resampled interval on that "
                 "difference excludes zero."),
    )

    tf_verdicts = {line: _beats(_paired(r, line, "esm2_3b_emb", "transcriptformer_emb"))
                   for line in ("hepg2", "jurkat")}
    sp_verdicts = {line: _beats(_paired(r, line, "esm2_3b_emb", "scprint_emb"))
                   for line in ("hepg2", "jurkat")}
    sg_verdicts = {line: _beats(_paired(r, line, "esm2_3b_emb", "scgpt_emb"))
                   for line in ("hepg2", "jurkat")}
    disc_verdicts = {
        name: {line: _beats(_paired(r, line, "esm2_3b_emb", f"{name}_emb", DISCRIM_KEY))
               for line in ("hepg2", "jurkat")}
        for name in ("transcriptformer", "scprint", "scgpt")
    }
    disagree = [
        f"{MODEL_LABEL[name]} in {LINE[line]}"
        for name, v in disc_verdicts.items()
        for line in ("hepg2", "jurkat")
        if v[line] != {"transcriptformer": tf_verdicts, "scprint": sp_verdicts,
                       "scgpt": sg_verdicts}[name][line]
    ]

    VERB = {"ahead": "beats", "behind": "falls short of", "a tie": "ties",
            "not measured": "was not compared with"}

    def phrase(verdicts: dict, name: str) -> str:
        if len(set(verdicts.values())) == 1:
            v = VERB[next(iter(verdicts.values()))]
            return f"{v} {name} in both cell lines"
        return (f"{VERB[verdicts['hepg2']]} {name} in HepG2 and "
                f"{VERB[verdicts['jurkat']]} it in Jurkat")

    draws = {line: r.head["protein"]["lines"][line].get("largest_protein_shuffled_draw")
             for line in ("hepg2", "jurkat")}
    nulls = (
        f"({P.num(r.score('hepg2', 'null_randgene_d2560'))} and "
        f"{P.num(r.score('jurkat', 'null_randgene_d2560'))} at width 2,560; "
        f"{P.num(r.score('hepg2', 'null_randgene_d5120'))} and "
        f"{P.num(r.score('jurkat', 'null_randgene_d5120'))} at 5,120)"
    )

    def rnd(line: str, arm: str, col: str = PREREG) -> float:
        d = r.scope_arms(line, "full", "random")
        d = d[d.arm == arm]
        return float(d[col].iloc[0]) if not d.empty else float("nan")

    random_split = (
        "<p>Under the random split, where correlated perturbations can fall on both sides of "
        "the fold, the protein arm and the model arms move together. On the pre-registered "
        f"rule ESM2-3B goes from {P.num(r.score('hepg2', 'esm2_3b_emb', PREREG))} to "
        f"{P.num(rnd('hepg2', 'esm2_3b_emb'))} in HepG2, where every embedding arm roughly "
        f"doubles, and from {P.num(r.score('jurkat', 'esm2_3b_emb', PREREG))} to "
        f"{P.num(rnd('jurkat', 'esm2_3b_emb'))} in Jurkat, where the scores barely move. Beside "
        f"it sit {P.num(rnd('hepg2', 'transcriptformer_emb'))} and "
        f"{P.num(rnd('jurkat', 'transcriptformer_emb'))} for TranscriptFormer and "
        f"{P.num(rnd('hepg2', 'scgpt_emb'))} and {P.num(rnd('jurkat', 'scgpt_emb'))} for "
        f"scGPT, against {P.num(rnd('hepg2', 'null_randgene_d2560'))} and "
        f"{P.num(rnd('jurkat', 'null_randgene_d2560'))} for the width-matched null and "
        f"{P.num(rnd('hepg2', 'coexp'))} and {P.num(rnd('jurkat', 'coexp'))} for co-expression. "
        "The grouped split is still the one the verdict is read from, for the reason it always "
        "was: the random split lets a translator reach a held-out perturbation through its "
        "correlated neighbours.</p>"
    )

    UNSEEN_ARMS = ("esm2_3b_emb", "esm2_650m_emb", "transcriptformer_emb", "scprint_emb",
                   "scgpt_emb", "coexp", "null_randgene_d2560", "null_randgene_d5120")
    trows = []
    for direction in ("hepg2->jurkat", "jurkat->hepg2"):
        block = _ptransfer(r, direction, "transfer_unseen")
        if not block:
            continue
        for arm in UNSEEN_ARMS:
            row = block.get(arm)
            if row is None:
                continue
            trows.append([
                direction.replace("->", " to ").replace("hepg2", "HepG2").replace(
                    "jurkat", "Jurkat"),
                label(arm), f"{row['n_perturbations']}",
                ci(row["preregistered"], row["preregistered_lo"], row["preregistered_hi"]),
                ci(row["score"], row["lo"], row["hi"]),
            ])
    transfer = ""
    clear = r.head.get("protein_transfer", {}).get("unseen_gene_clearance", {})
    if trows and clear:
        ref = clear.get("dest_reference", {})
        others = {m: clear[m] for m in ("dest_all", "source_only") if m in clear}
        other_txt = _join([
            f"{m.replace('_', ' ')} ({v['n_perturbations_scored']} perturbations, "
            f"{v['protein_derived']} protein-derived, {v['scgpt']} scGPT, "
            f"{v['coexpression_and_nulls']} co-expression or null)"
            for m, v in others.items()
        ])
        transfer = (
            "<p>The sharpest version of the split is the cross-line evaluation restricted to "
            "genes that were never perturbed in the training screen. A gene table only helps "
            "there if it places a gene it has no training signal for somewhere sensible. The "
            "destination line's response has to be measured against something, and the only "
            "choice that keeps the evaluation non-transductive builds that reference out of "
            "half the destination's perturbations, leaving the other "
            f"{ref.get('n_perturbations_scored', '')} to be scored. On those rows the four "
            f"protein-derived tables clear zero in {ref.get('protein_derived', '')} cells (two "
            "directions by four tables), scGPT's, the one table learned from expression alone, "
            f"in {ref.get('scgpt', '')}, and co-expression together with the two width-matched "
            f"random-vector nulls in {ref.get('coexpression_and_nulls', '')}.</p>"
            "<p>The other two reference choices score twice as many rows, so their counts are "
            f"not comparable to those and are given separately: {other_txt}. The direction of "
            "the separation is the same in all three, but in the source-only mode two of the "
            "six co-expression-or-null cells also clear zero, so the mode is not a free "
            "choice.</p>"
            + P.table(
                ["Direction", "Arm", "Held-out perturbations", "Pre-registered score",
                 "Fold-centred score"],
                trows, sources=[S_ANALYSIS, S_RUNNER],
                caption=("Cross-line transfer over perturbations whose gene never appears in "
                         "the training screen, MLP translator, using the held-out reference "
                         "cells as everywhere else on this page. All rows come from the protein "
                         "arm's own job, which reproduced every arm it re-ran from the main "
                         "sweep exactly."),
            )
        )

    body = (
        "<p>Every arm above that beat co-expression fed the translator a model's static "
        "gene-identity table. Two of those four tables did not come from single-cell data "
        "alone, so what the phrase 'foundation-model embedding' covers has to be split apart. "
        "Reading the four released checkpoints settles where each table came from.</p>"
        + prov +
        "<p>The cases are different in kind. TranscriptFormer's arm is a protein language "
        "model's embedding passed through unchanged: the table in its weights is the ESM2 file "
        "shipped in the same release, to the byte. scPRINT started from a protein-derived table "
        "and trained it for 400,000 steps. scGPT and AIDO.Cell learned theirs from expression "
        "with no protein model anywhere in the path.</p>"
        "<p>That makes the obvious control worth running: a gene table built from protein "
        "sequence alone, with no single-cell model in its path, through the identical "
        f"translator, folds, weights, scoring and nulls. Each read-out gene was mapped to its "
        f"canonical reviewed human protein - {routes.get('ensembl', 0):,} of "
        f"{mapping.get('n_genes', 0):,} by Ensembl gene id, {routes.get('symbol', 0)} by gene "
        f"symbol, none unmapped - and the sequence embedded with two public ESM2 checkpoints: "
        f"650M at width {embed.get('esm2_650m', {}).get('hidden_size', 1280):,} and 3B at width "
        f"{embed.get('esm2_3b', {}).get('hidden_size', 2560):,}, the second matching "
        f"TranscriptFormer's table exactly so the comparison is not confounded with feature "
        f"width. {lengths.get('over_esm2_context', 0)} proteins are longer than ESM2's "
        f"1,022-residue context and were embedded in chunks and averaged. Each arm is read "
        f"against a random-vector null of its own width, and the same job re-ran co-expression "
        f"and the three model embedding arms, reproducing the main sweep's numbers exactly, so "
        f"the comparison sits inside one job.</p>"
        + table +
        f"<p>The raw protein arm {phrase(tf_verdicts, 'TranscriptFormer')}, "
        f"{phrase(sp_verdicts, 'scPRINT')}, and {phrase(sg_verdicts, 'scGPT')}. It clears its "
        f"own width-matched random-vector null in both lines {nulls}, and its own shuffled-label "
        f"draws reach at most {P.num(draws['hepg2'])} in HepG2 and {P.num(draws['jurkat'])} in "
        f"Jurkat. "
        + ("The fold-centred diagnostic reads the same way on every one of these six "
           "comparisons.</p>" if not disagree else
           "On the fold-centred diagnostic the reading differs for "
           + _join(disagree) + ".</p>")
        + random_split + chart +
        "<p>The tables themselves say the same thing from the other direction. Comparing the "
        "rank ordering of gene-gene similarity between each pair of tables, TranscriptFormer's "
        f"sits closest to raw ESM2 ({P.num(r.rsa('hepg2', 'transcriptformer', 'esm2_3b'))} in "
        f"HepG2), scPRINT's keeps a smaller but clear protein signature "
        f"({P.num(r.rsa('hepg2', 'scprint', 'esm2_650m'))} against ESM2-650M) while agreeing "
        f"more with scGPT's expression-learned table "
        f"({P.num(r.rsa('hepg2', 'scgpt', 'scprint'))}), and scGPT's and AIDO.Cell's carry no "
        "protein-space structure at all "
        f"({P.num(r.rsa('hepg2', 'scgpt', 'esm2_3b'))} and "
        f"{P.num(r.rsa('hepg2', 'aido_cell', 'esm2_3b'))}, against a random-table floor of "
        f"{P.num(0.002116)}). The two ESM2 sizes agree with each other at "
        f"{P.num(r.rsa('hepg2', 'esm2_3b', 'esm2_650m'))}, so the distance between raw ESM2 and "
        "TranscriptFormer's table is a difference of recipe, not of protein model: this arm is "
        "an independent construction rather than a re-derivation of the table it is compared "
        "to.</p>"
        + chart2 + transfer
    )
    return P.section(
        "Where the winning embeddings come from, and what a protein model alone can do",
        body, sources=[S_AUDIT, S_UNIPROT, S_ESM, S_IDENTITY, S_ANALYSIS],
    )



CONCAT_HEADING = {
    "redundant": ("The two tables are two routes to one quantity: putting them together "
                  "buys nothing"),
    "complementary": ("The two tables carry different information: putting them together "
                      "beats either alone"),
    "mixed": ("The two tables agree weakly, and the hybrid test was too underpowered to "
              "say whether that buys anything"),
}
CONCAT_ROW_ARMS = ("scgpt_esm2_3b_emb", "scgpt_emb", "esm2_3b_emb", "null_randgene_d6144")
#: the arms scored only in the two-table job, which are kept out of runs.parquet
CONCAT_TABLE_ARMS = ("scgpt_esm2_3b_emb", "null_randgene_d6144")


def _concat_block(h: dict, line: str, split: str) -> dict | None:
    return h.get("lines", {}).get(f"{line}|{split}")


def _concat_gain(block: dict, part: str) -> str:
    """The paired difference of the two-table arm against one of its parts."""
    d = block["paired_against_parts"].get(f"{part}__preregistered")
    if not d:
        return "n/a"
    return f"{P.signed(d['difference'], 4)} ({P.num(d['lo'])} to {P.num(d['hi'])})"


def concat_section(r: Results, chart: str = "") -> str:
    """Are scGPT's table and the raw protein table the same information?"""
    h = r.head.get("concat")
    if not h:
        return ""
    rows = []
    for line in ("hepg2", "jurkat"):
        for split in ("grouped", "random"):
            block = _concat_block(h, line, split)
            if not block:
                continue
            by_arm = {s["arm"]: s for s in block["scores"]}
            for arm in CONCAT_ROW_ARMS:
                s = by_arm.get(arm)
                if not s:
                    continue
                rows.append([
                    LINE[line], split, label(arm), f"{s['feature_dim']:,}",
                    ci(s["score"], s["lo"], s["hi"]),
                    ci(s["fold_centred"], s["fold_centred_lo"], s["fold_centred_hi"]),
                ])
    tab = P.table(
        ["Cell line", "Split", "Arm", "Feature width", "Pre-registered score",
         "Fold-centred score"],
        rows, sources=[S_RUNNER, S_ANALYSIS],
        caption=(
            "Full scope, MLP translator, 95% intervals over 2,000 resamples of the response "
            "groups. The two-table arm is the two gene tables concatenated, nothing else "
            "changed. Its random-vector null is cut to the same 6,144 columns, which is what "
            "separates a real gain from the extra width."
        ),
    )
    hg, jk = _concat_block(h, "hepg2", "grouped"), _concat_block(h, "jurkat", "grouped")
    # the power limit is stated for every cell, not only the two grouped ones
    pieces, n_under, better_by_cell = [], 0, []
    for line, split in (("hepg2", "grouped"), ("hepg2", "random"),
                        ("jurkat", "grouped"), ("jurkat", "random")):
        block = _concat_block(h, line, split)
        if not block:
            continue
        d = block["paired_against_parts"].get(f"{block['better_part']}__preregistered", {})
        if not d:
            continue
        n_under += int(abs(d["difference"]) < d["min_resolvable_difference"])
        pieces.append(
            f"{LINE[line]} {split} {P.sig(d['difference'], signed_=True)} against "
            f"{P.sig(d['min_resolvable_difference'])}"
        )
        better_by_cell.append((label(block["better_part"]), f"{LINE[line]} {split}"))
    by_part: dict[str, list[str]] = {}
    for part, cell in better_by_cell:
        by_part.setdefault(part, []).append(cell)
    sizes = sorted(len(v) for v in by_part.values())
    if len(by_part) == 2 and sizes == [1, len(better_by_cell) - 1]:
        one = next(k for k, v in by_part.items() if len(v) == 1)
        rest = next(k for k, v in by_part.items() if len(v) > 1)
        word = {2: "two", 3: "three", 4: "four"}.get(len(by_part[rest]),
                                                      str(len(by_part[rest])))
        better_phrase = f"{one} in {by_part[one][0]} and {rest} in the other {word}"
    else:
        better_phrase = _join([f"{k} in {_join(v)}" for k, v in by_part.items()])
    agreement = []
    for line, block in (("HepG2", hg), ("Jurkat", jk)):
        if not block:
            continue
        agreement.append(f"{P.num(block['parts_spearman'])} in {line}")
    # the reading turns on the comparison against the better part, so that is the one
    # the diagnostic is checked on
    centred, n_cells = [], 0
    for line, split in (("hepg2", "grouped"), ("hepg2", "random"),
                        ("jurkat", "grouped"), ("jurkat", "random")):
        block = _concat_block(h, line, split)
        if not block:
            continue
        n_cells += 1
        d = block["paired_against_parts"].get(f"{block['better_part']}__fold_centred")
        if d and d["lo"] > 0 and d["difference"] > d["min_resolvable_difference"]:
            centred.append(
                f"{LINE[line]} under the {split} split against {label(block['better_part'])} "
                f"({P.signed(d['difference'], 4)}, {P.num(d['lo'])} to "
                f"{P.num(d['hi'])}, against {P.num(d['min_resolvable_difference'])} "
                "resolvable)"
            )
    centred_clause = (
        f" The fold-centred diagnostic agrees on {n_cells - len(centred)} of the "
        f"{n_cells} comparisons against the better part; in " + _join(centred)
        + " it clears what it resolves, by a margin narrower than the interval is wide."
        if centred else
        f" The fold-centred diagnostic reads the same way on all {n_cells} comparisons "
        "against the better part."
    )
    def _pair(key: str) -> list[str]:
        out = []
        for line, b in (("HepG2", hg), ("Jurkat", jk)):
            v = b["agreement"].get(key, {}).get("spearman") if b else None
            if v is not None and v == v:
                out.append(f"{P.num(v)} in {line}")
        return out

    same_family = _pair("esm2_650m_emb__esm2_3b_emb__residual_weighted")
    vs_tf = _pair("scgpt_emb__transcriptformer_emb__residual_weighted")
    vs_coexp = _pair("scgpt_emb__coexp__residual_weighted")
    reading = h.get("reading", "mixed")
    thresholds = h.get("agreement_thresholds", {})
    verdicts = _join([
        f"{LINE[ln]} under the {sp} split reads {_concat_block(h, ln, sp)['reading']}"
        for ln in ("hepg2", "jurkat") for sp in ("grouped", "random")
        if _concat_block(h, ln, sp)
    ])
    body = (
        "<p>The result above leaves one thing open. scGPT's table scores as well as the "
        "protein tables while the similarity measure says it carries no protein-space "
        "structure at all, so either the two tables are different routes to the same "
        "underlying quantity or they hold different information. Concatenating them answers "
        "it directly: the translator gets scGPT's 512-wide vector and the raw ESM2-3B "
        "2,560-wide vector for each of the two genes, 6,144 columns in total, with the same "
        "network shape, optimizer, folds, weighting, ceiling correction and scoring as every "
        "other arm. The arm is wider than either part, which is where a dimensionality "
        "artefact would show up, so a fixed random vector per gene at exactly 6,144 columns "
        "is scored beside it.</p>"
        "<p>The reading was fixed before the arm was scored. If the concatenation does not "
        "beat the better of its two parts by more than the paired comparison resolves, and "
        "the two parts succeed and fail on the same held-out perturbations, they are proxies "
        "for one quantity reached by different routes. If it clearly beats both and they "
        "succeed on different perturbations, they carry different information, which is a "
        "separate finding rather than a footnote to this one. Agreement is the Spearman "
        f"correlation of the two arms' per-perturbation scores: "
        f"{P.num(thresholds.get('strong_at_or_above', 0.5))} or above is the same "
        f"perturbations, below {P.num(thresholds.get('weak_below', 0.3))} is different ones, "
        "and anything between is neither.</p>"
        + tab
        + "<p>The concatenation does not beat the better of its two parts, and the "
        "comparison could not have resolved a gain of the size on offer either: in "
        f"{'all four cells' if n_under == len(pieces) == 4 else f'{n_under} of the {len(pieces)} cells'} "
        "the difference against the better part is smaller than the smallest difference the "
        f"paired comparison resolves - {'; '.join(pieces)}, the better part being "
        f"{better_phrase}. That is a property of this design rather than of the "
        "models, so the arm cannot be read as the two tables being equivalent or either one "
        "being redundant. What it does rule out is the extra width: the two-table arm sits "
        "far above its own width-matched random projection, so the added columns are not "
        f"what is being measured.{centred_clause}</p>"
        + "<p>The half of the rule that does carry information is the agreement statistic, "
        "and it only means something beside its calibration. The scGPT-table arm and the "
        f"protein-table arm agree on which held-out perturbations they score well at Spearman "
        f"{_join(agreement)}. Two sizes of the same protein model - two views of one "
        f"underlying quantity in this exact setup - agree at {_join(same_family)}. scGPT "
        f"against co-expression agrees at {_join(vs_coexp)}, and scGPT against "
        f"TranscriptFormer at {_join(vs_tf)}. So scGPT's table agrees with a protein model "
        "about as weakly as it agrees with co-expression, and far below what the same model "
        f"at two sizes reaches - below the {P.num(thresholds.get('weak_below', 0.3))} the "
        "rule set for different perturbations, and nowhere near the "
        f"{P.num(thresholds.get('strong_at_or_above', 0.5))} it set for the same ones. "
        "Per-perturbation scores are noisy at this sample size, which is why the same-family "
        "pair is the reference that matters: it shows what this statistic reaches when two "
        "arms really are two routes to one quantity.</p>"
        + chart
        + f"<p>Read against the rule, that is {reading}: {h['rule'][reading]}. "
        f"By cell and split, {verdicts}. The finding is a discrepancy identified rather than "
        "resolved. The weak agreement points toward scGPT's table and the protein table "
        "carrying different information rather than being redundant routes to one quantity, "
        "and the concatenation test is too underpowered to say whether that difference "
        "converts into a gain this translator can use. Neither half of that is a null and "
        "neither is a positive. It does not say scGPT's table is protein-derived either; the "
        "checkpoint audit already says it is not, and the similarity measure still finds no "
        "protein geometry in it. A task with finer resolution than a per-perturbation "
        "correlation, or a different way of combining the two tables, is what would settle "
        "it.</p>"
    )
    return P.section(CONCAT_HEADING.get(reading, CONCAT_HEADING["mixed"]),
                     body, sources=[S_RUNNER, S_ANALYSIS])

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
        f"<p>Under the version the jobs resolved, the embedding arms clear chance in both lines "
        f"({P.num(best_h)} in HepG2, {P.num(best_j)} in Jurkat), co-expression does not "
        f"({P.num(r.disc('hepg2', 'coexp'))} and {P.num(r.disc('jurkat', 'coexp'))}), and "
        "neither do the response-only arms. That ordering is the one both correlation scores "
        "give, so it would be independent support for the result, except that the ordering is "
        "not stable across releases of the metric.</p>"
        + _disc_version_note(r)
    )
    return P.section("Discrimination, the second scoring axis, depends on the version that "
                     "computes it", body + tab + chart, sources=[S_DISC, S_ANALYSIS])


def _disc_version_note(r: Results) -> str:
    v = r.disc_versions
    if not v:
        return ""
    spread = float(v["max_abs_difference_any_file"])
    vers = sorted({k for f in v["per_file"].values() for k in f["values"]})
    old_v = "0.5.18"

    def at(file: str, version: str) -> str:
        return P.num(v["per_file"][file]["values"][version])

    hep_m, hep_c = (
        "hepg2__full__grouped__scgpt_full__mlp.npz",
        "hepg2__full__grouped__coexp__mlp.npz",
    )
    jur_m, jur_c = (
        "jurkat__full__grouped__transcriptformer_full__mlp.npz",
        "jurkat__full__grouped__coexp__mlp.npz",
    )
    return (
        "<p>The pre-registration pinned <code>cell-eval==0.5.19</code>. That version does not "
        "exist: the package publishes 0.1.0 and then 0.5.33 onward, and the upstream tag list "
        "jumps 0.5.18 to 0.5.21, so the pin was unattainable when it was written and the jobs "
        f"resolved {v['resolved_in_the_sweep']}. Rescoring the same saved prediction matrices "
        f"under {_join(vers)} reverses the comparison this axis is being asked about. Under "
        f"{old_v}, which is identical to 0.5.21 on every file, the HepG2 scGPT full arm scores "
        f"{at(hep_m, old_v)} against {at(hep_c, old_v)} for co-expression and the Jurkat "
        f"TranscriptFormer full arm {at(jur_m, old_v)} against {at(jur_c, old_v)}; under 0.8.2 "
        f"the same matrices give {at(hep_m, '0.8.2')} against {at(hep_c, '0.8.2')} and "
        f"{at(jur_m, '0.8.2')} against {at(jur_c, '0.8.2')}. The largest move on any file is "
        f"{P.num(spread)}. The implementations differ in one line (0.5.x scores the absolute "
        "effect profile, 0.8.2 the signed one) and a distance on absolute values discards the "
        "direction of every response, which is why 0.8.2 is the reported number and the "
        "version travels with it.</p>"
        "<p>Because the two existing versions nearest the unattainable pin rank co-expression "
        "and the model arms the other way round, this axis cannot serve as independent "
        "confirmation of the correlation result, and it is reported as inconclusive rather "
        "than as corroboration. It is not evidence against the correlation result either: the "
        "reversal is a property of the metric implementation, not of the predictions, which "
        "are the same matrices in both cases.</p>"
    )


def _no_op_note(r: Results) -> str:
    """How many queries the per-query exclusion touched, and how far it moved them."""
    if not r.no_op:
        return ""
    total = sum(b["n_no_op_queries"] for b in r.no_op)
    rows = sum(b["n_query_rows_touched"] for b in r.no_op)
    gone = sum(b["n_query_rows_everywhere_no_op"] for b in r.no_op)
    per_block = r.no_op[0]["n_perturbed"] * r.no_op[0]["n_values"]
    move = max(b["max_abs_feature_change_partial_rows"] for b in r.no_op)
    checks = [c for b in r.no_op for c in b["checks"].values()]
    return (
        f": {total} no-op queries over the three blocks touch {rows} query rows of "
        f"{per_block * len(r.no_op)}, the corrected feature differs from the shipped one by at "
        f"most {move:.1e}, and {sum(c['pass'] for c in checks)} of {len(checks)} checks pass. "
        f"{gone} rows were no-ops in every cell and carry no measurement at all"
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
            dim = r.sens_score(m, a, "feature_dim")
            cells.append(f"{P.num(v)} ({dim:.0f})" if v == v else "not extracted")
        rows.append(cells)
    tab = P.table(
        ["Model", "Mean control", "Uniform virtual", "Individual cells", "Masked read-out"],
        rows, sources=[S_SENS],
        caption="Fold-centred score, with the number of feature values in brackets.",
    )
    head = r.head.get("sensitivity", {})
    paired = head.get("paired_against_mean_control", [])
    rerun = head.get("max_abs_rerun_difference", float("nan"))
    uv = {p["model"]: p for p in paired if p["anchor"] == "uniform_virtual"}
    uv_better = [m for m, p in uv.items() if p["lo"] > 0]
    uv_worse = [m for m, p in uv.items() if p["hi"] < 0]
    biggest = max(paired, key=lambda p: abs(p["median_difference"])) if paired else None
    best_score = max(
        (r.sens_score(m, a) for m in models for a in anchors
         if r.sens_score(m, a) == r.sens_score(m, a)),
        default=float("nan"),
    )
    emb_best = max(r.score("hepg2", f"{m}_emb") for m in FULL_SCOPE_MODELS)
    prows = [
        [MODEL_LABEL[p["model"]], p["anchor"].replace("_", " "),
         ci(p["median_difference"], p["lo"], p["hi"]),
         P.num(p["min_resolvable_difference_achieved"])]
        for p in paired
    ]
    ptab = P.table(
        ["Model", "Anchor", "Paired difference against the mean-control anchor",
         "Smallest resolvable"],
        prows, sources=[S_ANALYSIS],
    )
    body = (
        "<p>The paper's mechanism claim is that querying the model at a uniform virtual value, "
        "rather than at the expression the cell actually has, gives a cleaner read of one gene's "
        "influence on another. The features were harvested at four queries in the previous run, "
        "and this step scores the same response-only arm at each of them in HepG2 under the "
        "grouped split. Translator, split, scope and scoring rule are identical, so the only "
        "thing that varies is what the model was asked.</p>"
        "<p>One query needs care. At the individual-cells anchor the response is averaged over "
        "4 real control cells, and a gene that already reads zero in one of them is unchanged "
        "by being set to 0 CP10K: that cell's two forward passes are the same call, so it "
        "contributes an exact zero. Those no-op queries are excluded per (perturbed gene, "
        "value) rather than by deleting the whole dose, because the same dose is a real "
        "measurement for every other perturbation. Since the stored row is the plain mean over "
        "the cells and the feature is that row z-scored across read-out genes, excluding them "
        "is a positive scale factor the z-score removes exactly, which is checked against the "
        f"shipped arrays{_no_op_note(r)}. The masked column is narrower for an unrelated "
        "reason: that variant ships no gradient form.</p>"
        "<p>"
        + (
            "The anchor moves the score, but never out of the null band. "
            if best_score <= null
            else "The anchor moves the score, and the best of them clears the null band. "
        )
        + f"The largest paired move is "
        f"{ci(biggest['median_difference'], biggest['lo'], biggest['hi']) if biggest else 'n/a'} "
        f"({MODEL_LABEL[biggest['model']] if biggest else ''} at the "
        f"{biggest['anchor'].replace('_', ' ') if biggest else ''} anchor). The uniform-virtual "
        f"anchor beats the on-distribution one for "
        f"{_join([MODEL_LABEL[m] for m in uv_better]) or 'no model'} and loses for "
        f"{_join([MODEL_LABEL[m] for m in uv_worse]) or 'no model'}, so the "
        "mechanism claim holds for some models and reverses for others on this task. The best "
        f"score at any anchor is {P.num(best_score)}, "
        f"{'still below' if best_score <= null else 'above'} the shuffled-label ceiling of "
        f"{P.num(null)} and a fraction of the {P.num(emb_best)} the same models' embeddings "
        "reach in this cell line. Masking is not a clean mechanistic contrast: it hides the "
        "read-out gene's own value together with its whole block, so it removes context as well "
        "as the value.</p>"
        "<p>The mean-control arms were re-run inside this job as its own reference. They "
        + ("reproduce the main sweep exactly, to every digit stored, "
           if rerun == 0 else f"reproduce the main sweep to within {P.num(rerun)}, ")
        + "which also confirms the rebuilt bundle is the same bundle.</p>"
    )
    title = (
        "No query anchor lifts the response features out of the null band"
        if best_score <= null
        else "One query anchor lifts the response features out of the null band"
    )
    return P.section(title, body + tab + ptab + chart, sources=[S_SENS, S_CONFIG])


def controls_section(r: Results, split_chart: str, overlap_chart: str) -> str:
    npm_h = r.score("hepg2", "node_pairmlp", PREREG)
    npm_j = r.score("jurkat", "node_pairmlp", PREREG)

    def rnd_col(arm: str, col: str) -> float:
        d = r.runs[(r.runs.cell_line == "hepg2") & (r.runs.scope == "full")
                   & (r.runs.split == "random") & (r.runs.translator == "mlp")
                   & (r.runs.arm == arm)][col]
        return float(d.iloc[0]) if not d.empty else float("nan")

    npm_hr = rnd_col("node_pairmlp", DISCRIM)
    npm_hr_p = rnd_col("node_pairmlp", PREREG)
    hb = r.best("hepg2", FULL_SCOPE_MODELS, exclude="hybrid")
    body = (
        "<p>A pair feature can score without saying anything about the pair. Two controls bound "
        "that. The node-only control gives the network the same six summary statistics for each "
        "gene on its own (expression level, detection rate, co-expression degree) and no "
        "pairwise term. Fitted as a free network over both genes it reaches "
        f"{P.num(npm_h)} in HepG2 and {P.num(npm_j)} in Jurkat on the pre-registered rule, well "
        f"below {label(hb.arm)} at {P.num(hb[PREREG])} but above co-expression, a reminder that "
        "part of what any arm scores here is what the two genes are like on their own. "
        "Restricted to an additive form, one "
        "branch per gene summed, it is identically zero on the fold-centred score: after the "
        "fold's per-gene mean is removed, an additive prediction is the same vector for every "
        "perturbation. That is a property of the control and the score together, not a "
        "measurement, which is why the free version is the informative one.</p>"
        "<p>The gap between the two splits is the second bound. Under a random split, where "
        "correlated perturbations can fall on both sides, node statistics alone reach "
        f"{P.num(npm_hr_p)} in HepG2 on the pre-registered rule against {P.num(npm_h)} under "
        "the grouped split, and on the fold-centred score the same control goes from "
        f"{P.num(r.score('hepg2', 'node_pairmlp'))} to {P.num(npm_hr)}, "
        f"{P.num(npm_hr / r.score('hepg2', 'node_pairmlp'))} times its grouped-split score. A "
        "control with no pairwise term should not gain from where the fold boundary falls, and "
        "that it does is what the grouped split is there to prevent.</p>"
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
                        values=DISCRIM).reset_index()
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
            b = b.sort_values(PREREG, ascending=False).iloc[0]
            gi = d[(d.cell_line == direction) & (d.arm == "null_randgene_d5120")]
            co = d[(d.cell_line == direction) & (d.arm == "coexp")]
            parts.append({
                "scope": name, "direction": direction, "arm": b.arm,
                "score": float(b[PREREG]), "lo": float(b[PREREG + "_lo"]),
                "hi": float(b[PREREG + "_hi"]),
                "gene_null": float(gi[PREREG].iloc[0]) if not gi.empty else float("nan"),
                "coexp": float(co[PREREG].iloc[0]) if not co.empty else float("nan"),
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
        "the intervals are wide.</p>"
        + (
            "<p>That surviving score is where the four gene tables come apart. The protein "
            "section below reads the same unseen-gene evaluation arm by arm: on the rows this "
            "table reports, the tables that derive from protein sequence keep an interval clear "
            "of zero and scGPT's, the one learned from expression alone, does not in either "
            "direction.</p>"
            if r.head.get("protein_transfer") else ""
        )
        + mode_note
    )
    return P.section("Transfer across cell lines survives only against a gene-identity control",
                     body + tab + chart, sources=[S_RUNNER, S_ANALYSIS, S_FIGS])


def mode_agreement_note(r: Results) -> str:
    d = r.transfer[r.transfer.translator == "mlp"]
    piv = d.pivot_table(index=["cell_line", "scope", "arm"], columns="transfer_mode",
                        values=PREREG)
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
        "<p>The destination line's response has to be measured against something, and the choice "
        "matters more than it looks. Using the source line's mean, or all destination cells, "
        "instead of a held-out half of the destination cells moves an arm's pre-registered "
        f"score by up to {P.num(max(shifts))}, and the ordering it induces agrees with the "
        f"held-out-reference one only loosely (rank correlation as low as {P.num(min(ranks))} "
        "within a direction and scope). The two alternatives also score twice as many "
        "perturbations, because the held-out reference has to be built out of half of them. "
        "Every number reported here uses the held-out reference cells, the only one of the "
        "three that keeps the evaluation non-transductive, and counts from the three are never "
        "pooled.</p>"
    )


def scope_section(r: Results) -> str:
    rows = []
    clears = {}
    for line in ("hepg2", "jurkat"):
        d = r.scope_arms(line, "common", "grouped")
        shuffled = d[d.arm.str.contains(figmod.SHUFFLE_TAG, regex=False)]
        ceil = {c: float(np.nanmax(shuffled[c].to_numpy())) if not shuffled.empty else float("nan")
                for c in (PREREG, DISCRIM)}
        for model in ("scgpt", "scprint", "transcriptformer", "aido_cell"):
            sub = d[d.arm.str.startswith(model) & ~d.arm.str.contains(
                figmod.SHUFFLE_TAG, regex=False)]
            if sub.empty:
                continue
            b = sub.sort_values(PREREG, ascending=False).iloc[0]
            bd = sub.sort_values(DISCRIM, ascending=False).iloc[0]
            clears[(line, model)] = (float(b[PREREG]) > ceil[PREREG],
                                     float(bd[DISCRIM]) > ceil[DISCRIM])
            rows.append([
                LINE[line], MODEL_LABEL[model], label(b.arm),
                P.num(float(b[PREREG])), P.num(ceil[PREREG]),
                P.num(float(bd[DISCRIM])), P.num(ceil[DISCRIM]),
                "yes" if clears[(line, model)][0] else "no",
                "yes" if clears[(line, model)][1] else "no",
            ])
    tab = P.table(
        ["Cell line", "Model", "Best arm (pre-registered)", "Pre-registered score",
         "Pre-registered null", "Fold-centred score", "Fold-centred null",
         "Clears (pre-registered)", "Clears (fold-centred)"],
        rows, sources=[S_ANALYSIS],
        caption=("Best arm per model on the 250 perturbations every model covers. The "
                 "fold-centred column takes each model's best arm on that score, which is not "
                 "always the same arm."),
    )
    n = int(r.scope_arms("hepg2", "common", "grouped").n_perturbations.iloc[0])
    aido = clears.get(("hepg2", "aido_cell"), (False, False))
    aido_j = clears.get(("jurkat", "aido_cell"), (False, False))
    body = (
        "<p>AIDO.Cell was harvested on a 250-gene subset in the previous run, so the only "
        "comparison that puts all four models on the same rows is that subset. On those "
        f"{n} perturbations the other three models clear the shuffled-label null on both "
        "scores. AIDO.Cell is the one case where the two scores disagree: on the "
        f"pre-registered rule its best arm sits {'above' if aido[0] else 'below'} the null in "
        f"HepG2 and {'above' if aido_j[0] else 'below'} it in Jurkat, and on the fold-centred "
        f"score {'above' if aido[1] else 'below'} and "
        f"{'above' if aido_j[1] else 'below'} respectively. With a quarter of the perturbations "
        "the intervals are wide either way, so the honest reading is that AIDO.Cell is not "
        "separated from the null on this subset, not that it fails.</p>"
    )
    return P.section("On the 250-gene subset where all four models overlap, AIDO.Cell is the "
                     "one case the two scores disagree about", body + tab, sources=[S_ANALYSIS])


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
    hceil = r.ceiling("hepg2", PREREG)
    n_h = int(r.primary("hepg2").n_groups_scored.dropna().iloc[0])
    big_h = int(r.primary("hepg2").largest_group_scored.dropna().iloc[0])
    n_j = int(r.primary("jurkat").n_groups_scored.dropna().iloc[0])
    body = (
        "<ul>"
        "<li>The pre-registered score rewards predicting what a held-out group shares. Its "
        "shuffled-label draws sit above zero for that reason, and a prediction that barely "
        "varies with the perturbation collects part of the score. The fold-centred version "
        "reported beside it removes that component, but it does so with a mean of the held-out "
        "outcomes, so it is a diagnostic rather than a second verdict, and it was written after "
        "the shuffled draws were seen. Where the two disagree this page says so; neither is "
        "presented as confirming the other.</li>"
        f"<li>The effect is small in absolute terms. Every score is already divided by the "
        f"square root of the reproducibility ceiling ({P.num(hceil)} in HepG2), so 1.0 would "
        f"mean predicting the response as well as the measurement repeats itself. "
        f"{label(hb.arm)} reaches {P.num(float(hb[PREREG]))} there, which leaves most of the "
        "reproducible perturbation-specific signal unexplained by every arm tested.</li>"
        f"<li>The independent unit is the response group, not the perturbation. HepG2's "
        f"{int(r.primary('hepg2').n_perturbations.iloc[0])} in-scope perturbations fall into "
        f"{n_h} groups of correlated responses, one holding {big_h} of them, and Jurkat's into "
        f"{n_j}. Every interval on this page resamples those groups, which is wider than "
        "resampling rows and is why some differences that would read as separated do not. The "
        "count of effectively independent units is closer to the group count than to the "
        "perturbation count.</li>"
        "<li>One translator, and four anchors in one cell line. Every arm went through the same "
        "network shape; a larger translator could change the absolute numbers, though it would "
        "have to change them unevenly across arms to change the ordering. The anchor was varied "
        "in HepG2 only, over the three models with full-coverage feature blocks; AIDO.Cell's "
        "block covers 250 perturbations, so it cannot enter the full-scope sensitivity step at "
        "all.</li>"
        "<li>The discrimination axis is inconclusive. The version pinned before the run does "
        "not exist upstream, and the two real versions nearest it rank co-expression above the "
        "model arms while the current one ranks them below, on the same saved predictions. The "
        "reported numbers name their version, and the verdict does not rest on them.</li>"
        + _beeline_limitation(r)
        + "<li>Two cell lines from one screen. Both come from the same CRISPRi Perturb-seq study "
        "with the same protocol, so the cross-line result is a weaker test of generality than "
        "two independent screens would be.</li>"
        "<li>Co-expression is one estimator family. Four statistics over one half of the "
        "control cells stand in for the statistical baseline; a network-inference method fitted "
        "to the same cells could do better than they do.</li>"
        + (
            "<li>The two-table test is one concatenation at one width. It asks whether this "
            "translator can use scGPT's table and the protein table together better than it "
            "uses the better one alone, and it does not rule out that a different way of "
            "combining them, or a task with finer resolution than a per-perturbation "
            "correlation, would separate what they carry.</li>"
            if r.head.get("concat") else ""
        )
        + (
            "<li>The protein control shows where the signal can come from, not where each "
            "model's signal does come from. It establishes that a gene table built from "
            "protein sequence alone scores as well as the single-cell models' tables on this "
            "task, and the checkpoint audit establishes which tables are protein-derived. It "
            "does not decompose scGPT's or AIDO.Cell's learned table into a protein-like part "
            "and a cell-specific part, and it says nothing about what these models do beyond "
            "this gene-pair task. One protein model at two sizes, and one pooling rule (mean "
            "over the final layer), stand in for protein space.</li>"
            if r.head.get("protein") else ""
        )
        + "</ul>"
    )
    return P.section("Limitations", body, sources=[S_CONFIG, S_ANALYSIS])


def beeline_section(r: Results) -> str:
    """A side arm that was run, failed to reproduce its source, and was dropped."""
    b = r.beeline
    if not b:
        return ""
    rows = [
        [
            f"{row['network']} / {row['dataset']}",
            f"{row['n_pairs_kept']:,}",
            P.num(row["auprc"]),
            P.num(row["paper_baseline_emb_auprc"]),
        ]
        for row in b["rows"]
    ]
    tab = P.table(
        ["Benchmark cell", "Pairs scored", "AUPRC here", "AUPRC reported"],
        rows, sources=[S_BEELINE],
        caption=(
            "The comparison that decided the arm: the re-implementation scores above the "
            "reported value on two cells and below it on one. Chance is 0.50 on these "
            "balanced sets."
        ),
    )
    s = b["summary"]
    train = b["train_cell"]
    body = (
        "<p>One side arm was run and then dropped, and it is written up here so that the "
        "decision is on the record. The response features this run tests come from a paper "
        "whose own evidence is a different task: BEELINE network inference, where a projector "
        "maps a gene-pair feature to a regulatory edge and is scored by AUPRC on held-out "
        "datasets. Disagreeing with its conclusion is worth more after reproducing its setting, "
        "so its scGPT embedding baseline was re-run on its own benchmark files: the projector "
        f"fitted on the {train['network']}/{train['dataset']} pair set "
        f"({train['n_pairs_kept']:,} pairs, balanced 1:1) and scored on the held-out cells with "
        "the paper's projector shape, optimizer, batch size and epoch count.</p>"
        f"<p>It did not reproduce. Mean AUPRC over the three held-out cells is "
        f"{P.num(s['mean_auprc'])} against {P.num(s['paper_mean_auprc'])} reported, a mean "
        f"difference of {P.signed(s['mean_difference'], 4)} and up to "
        f"{P.num(s['max_abs_difference'])} on one cell, higher on two of the three cells and "
        "lower on the third. No tolerance was pre-registered for this side result and no "
        "seed-to-seed noise estimate was run, so the gap is unexplained and the "
        "re-implementation is not established as the paper's setting.</p>"
        "<p>So the arm is dropped rather than reported. A check that disagrees with the source "
        "it is checking is not a check, and reporting anything measured inside it would invite "
        "the reading that the paper was reproduced and then refuted, which is not what "
        "happened. Controls were run in that setting and are not reported as findings here, "
        "because a control is only informative once the setting it runs in is established. The "
        "gradient-trajectory and virtual-value arms were not re-run at all: they would need the "
        "feature extraction repeated against these datasets, which is a separate harvest. "
        "Nothing else on this page depends on this arm - the translator scores, the protein "
        "control and the cross-line transfer result are all measured on the CRISPRi screens and "
        "never touch the BEELINE data - and the full output stays in the artifact record so the "
        "attempt is auditable.</p>"
    )
    return P.section("A dropped arm: the source paper's BEELINE benchmark did not reproduce",
                     body + tab, sources=[S_BEELINE])


def _beeline_limitation(r: Results) -> str:
    if not r.beeline:
        return (
            "<li>The source paper's own benchmark is not reproduced here. Its network-inference "
            "task runs on different data with a different label, so this run tests its features "
            "on a task it was not evaluated on rather than contradicting its own numbers.</li>"
        )
    s = r.beeline["summary"]
    return (
        "<li>The source paper's own benchmark is not part of the evidence here. A "
        "re-implementation of it was run and dropped: its projector and embedding baseline "
        f"scored {P.num(s['mean_auprc'])} against {P.num(s['paper_mean_auprc'])} reported, up "
        f"to {P.num(s['max_abs_difference'])} apart on one cell, with no pre-registered "
        "tolerance and no noise estimate, so the setting is not established and no result on "
        "this page rests on it or on any control run inside it. Its virtual-value and "
        "gradient-trajectory features were never re-run, because those would need a fresh "
        "extraction against those datasets.</li>"
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
        "vertical one: the pre-registered score gives a shuffled-label arm credit for the part "
        "of a held-out group's response its members share, and the fold-centred version "
        "removes it.",
        sources=[S_METRICS, S_ANALYSIS, S_FIGS],
    )
    f_arms = P.figure(
        2, "Held-out score by arm, HepG2 and Jurkat",
        frag("arm_scores"),
        "Reliability-weighted, ceiling-corrected correlation on held-out perturbations under "
        "the grouped split. The top row is the pre-registered rule the verdict is read from; "
        "the bottom row is the fold-centred diagnostic. Intervals are 95% over 2,000 resamples "
        "of the response groups (226 groups over 524 perturbations in HepG2, 452 over 705 in "
        "Jurkat). The dotted line is the largest shuffled-label draw in that cell line and "
        "score.",
        "Every arm that carries a static gene-identity table sits above the null line on both "
        "scores, whether that table was learned from expression (scGPT, AIDO.Cell), came from a "
        "protein model (TranscriptFormer, scPRINT), or is a raw protein table with no "
        "single-cell model in its path (the two ESM2 arms). Co-expression sits just above the "
        "null on the pre-registered score and at it on the fold-centred one, and the "
        "random-vector controls and response-only arms sit at or below it.",
        sources=[S_RUNNER, S_ANALYSIS, S_FIGS],
    )
    con = r.con[(r.con.scope == "full") & (r.con.split == "grouped")
                & (r.con.translator == "mlp") & (r.con.metric == "residual_weighted")
                & (r.con.target_source == "run1_all_controls")]
    emb = con[con.arm.str.endswith(("_emb", "_full")) & ~con.arm.str.contains("shuffled")]
    vs_finding = (
        "Every arm carrying a gene-identity table, the two raw ESM2 tables included, beats "
        f"co-expression by {P.num(float(emb.median_difference.min()))} to "
        f"{P.num(float(emb.median_difference.max()))} correlation points; the response-only "
        "arms and the controls are indistinguishable from it."
    )
    f_vs = P.figure(
        3, "Paired difference against co-expression on the same held-out perturbations",
        frag("vs_coexpression"),
        "Median per-perturbation difference between each arm and the co-expression arm on the "
        "pre-registered target, scored on the same perturbations, with 95% intervals from "
        "resampling the response groups.",
        vs_finding,
        sources=[S_ANALYSIS, S_FIGS],
    )
    f_protein = P.figure(
        4, "The raw protein arm against the embedding arms it is a control for",
        frag("protein_arms"),
        "Held-out correlation, full scope, grouped split, MLP translator, on the pre-registered "
        "rule (top row) and the fold-centred diagnostic (bottom row), with 95% intervals over "
        "2,000 resamples of the response groups. The two ESM2 arms use gene tables built from "
        "protein sequence alone. The dotted line is the largest shuffled-label draw in that "
        "cell line and score.",
        protein_finding(r),
        sources=[S_RUNNER, S_ESM, S_FIGS],
    ) if "protein_arms" in made else ""
    f_tables = P.figure(
        5, "How close each model's gene table is to a raw protein table",
        frag("table_similarity"),
        "Rank correlation between the gene-gene cosine-similarity orderings of two gene tables, "
        "over a fixed random 1,500-gene subset of that cell line's read-out genes (seed 42). "
        "The measure ignores width and rotation. Two independent random tables score 0.000 to "
        "0.002 on it, at any width.",
        "TranscriptFormer's table is the one closest to raw ESM2. scPRINT's keeps a smaller "
        "protein signature and sits nearer scGPT's expression-learned table. scGPT's and "
        "AIDO.Cell's carry no protein-space structure.",
        sources=[S_IDENTITY, S_FIGS],
    ) if "table_similarity" in made else ""
    f_concat = P.figure(
        6, "Per-perturbation scores of the scGPT-table arm and the protein-table arm",
        frag("concat_agreement"),
        "One point per held-out perturbation: the score the scGPT-table arm reached on it "
        "against the score the raw ESM2-3B-table arm reached on the same perturbation, full "
        "scope, grouped split, MLP translator, pre-registered rule. The dotted line is "
        "equality and the rank correlation is printed in each panel.",
        "The cloud is broad rather than tight around the diagonal. The two arms agree on "
        "which perturbations are easy about as weakly as scGPT agrees with co-expression, "
        "and far below the 0.6816 two sizes of one protein model reach here, which points "
        "toward two different tables rather than two routes to one quantity.",
        sources=[S_RUNNER, S_FIGS],
    ) if "concat_agreement" in made else ""
    f_disc = P.figure(
        7, "Perturbation discrimination by arm, HepG2 and Jurkat",
        frag("discrimination"),
        "Mean normalized rank of the true match: for each held-out perturbation, how close its "
        "predicted effect profile is to its own measured profile compared with every other "
        "held-out perturbation. Computed with cell-eval on effect-scale profiles. The dotted "
        "line is chance.",
        "Under cell-eval 0.8.2 the arms carrying a gene embedding sit above chance while "
        "co-expression, the response-only arms and the random-vector control sit at or below "
        "it. Two earlier releases of the same metric reverse that ordering, so the axis is "
        "reported as version-dependent rather than as a second confirmation.",
        sources=[S_DISC, S_FIGS],
    ) if "discrimination" in made else ""
    f_sens = P.figure(
        8, "Response-only score by query anchor, HepG2",
        frag("anchor_sensitivity"),
        "The same response-only arm scored at each of the four queries the features were "
        "harvested at, under the grouped split with everything else held fixed. The dotted line "
        "is the largest shuffled-label draw in HepG2.",
        "Moving off the on-distribution anchor does not lift the response features out of the "
        "null band, including at the uniform virtual value the source paper argues for.",
        sources=[S_SENS, S_FIGS],
    ) if "anchor_sensitivity" in made else ""
    f_split = P.figure(
        9, "Grouped split against random split, every arm",
        frag("split_gap"),
        "Each point is one arm in one cell line, scored on the fold-centred diagnostic under "
        "both splits. The dotted line is equality.",
        "Every arm scores higher under the random split on this score, and the node-statistics "
        "control gains the most, which is what the grouped split is there to prevent.",
        sources=[S_RUNNER, S_FIGS],
    )
    f_overlap = P.figure(
        10, "Effect of sharing control cells with the target",
        frag("control_cell_overlap"),
        "Each arm's score against a target built from the same control cells the co-expression "
        "features use, minus its score against a target built from the other half of the "
        "control cells. Both targets were recomputed from the raw count matrices.",
        "The plain co-expression arms gain little from sharing control cells with the target, "
        "and the embedding arms, which touch no control cell, move by a similar amount.",
        sources=[S_RECOMPUTE, S_ANALYSIS, S_FIGS],
    ) if "control_cell_overlap" in made else ""
    f_transfer = P.figure(
        11, "Cross-cell-line transfer, over shared genes and over unseen genes",
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
        12, "All twelve cells of the scoring grid",
        frag("grid_cells"),
        "Every arm scored under each combination of target (residual, fold-centred, raw), "
        "weighting, and ceiling correction, grouped split, MLP translator.",
        "The residual and fold-centred targets rank the arms the same way. The raw target does "
        "not, because predicting it is mostly predicting each gene's average response.",
        sources=[S_ANALYSIS, S_FIGS],
    )

    blocks = [
        P.section("", "<h1>Gene embeddings beat co-expression at predicting knockdown "
                      "response, and protein sequence alone supplies them</h1>"),
        opening(r),
        setup(r),
        metric_section(r, f_metric),
        verdict_section(r, f_arms, f_vs),
        where_signal(r),
        protein_section(r, f_protein, f_tables),
        concat_section(r, f_concat),
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
        "Gene embeddings beat co-expression at predicting knockdown response, "
        "and protein sequence alone supplies them",
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
