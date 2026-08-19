"""Assemble the data-preparation report from the artifacts themselves.

Every number in the report is read out of a file that ships with it, so the
report cannot drift from the artifacts. The primary output is a single
self-contained HTML page with every figure embedded inline; a Markdown copy of
the same content is written beside it for reading in the repository.

    uv run python -m scfm_pfm.report.build --root results/artifacts_mirror

The page is composed from semantic HTML only: the Lab injects the stylesheet, so
this module never writes colours, sizes, or inline styles.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from silico_figures import PLOTLY_CONFIG

from .. import config as C
from .. import paths
from ..data import ceilings_grid
from ..data import sources as data_sources
from . import figures as figmod

LINES = ("hepg2", "jurkat")
LABEL = {"hepg2": "HepG2", "jurkat": "Jurkat"}
MODELS = ("scgpt", "aido_cell", "scprint", "transcriptformer")
TITLE = "Replogle-Nadig effect matrices and the four-model response contract"


# --------------------------------------------------------------------------
# tiny document model: the same blocks render to HTML and to Markdown
# --------------------------------------------------------------------------


@dataclass
class Head:
    level: int
    text: str


@dataclass
class Para:
    text: str


@dataclass
class Bullets:
    items: list[str]


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]


@dataclass
class Code:
    text: str


@dataclass
class FigRef:
    name: str


@dataclass
class Section:
    blocks: list = field(default_factory=list)
    sources: str = ""


def _esc(text: str) -> str:
    return html.escape(str(text), quote=False)


def _inline(text: str) -> str:
    """Markdown-ish inline markup to HTML: **strong** and `code`."""

    out = _esc(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`([^`]+?)`", r"<code>\1</code>", out)
    return out


def _first_sentence(text: str) -> str:
    """First sentence, without cutting at an abbreviation such as 'i.e.'."""

    parts = re.split(r"(?<!\bi\.e)(?<!\be\.g)(?<!\bcf)\.\s+", str(text).strip())
    return parts[0].rstrip(".") if parts else str(text)


def _sig(x: float, n: int = 4) -> str:
    """Four significant figures, switching to scientific notation for small values."""

    try:
        v = float(x)
    except (TypeError, ValueError):
        return "?"
    if v != v:
        return "n/a"
    if v != 0 and abs(v) < 1e-3:
        return f"{v:.{n - 1}e}"
    return f"{v:.{n}g}"


def _fmt(x: float, n: int = 4) -> str:
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return "n/a"


def _signed(x: float) -> str:
    try:
        return f"{float(x):+.4e}"
    except (TypeError, ValueError):
        return "n/a"


# --------------------------------------------------------------------------
# figure captions: title, what it plots, the takeaway, and its sources
# --------------------------------------------------------------------------


def figure_text(ctx: dict, rel: str) -> dict[str, dict[str, str]]:
    src = lambda name, extra: (  # noqa: E731
        f"{rel}/src/scfm_pfm/report/figures.py;{extra}"
    )
    prep = f"{rel}/src/scfm_pfm/data/prepare.py"
    ceil = f"{rel}/src/scfm_pfm/data/ceilings.py"
    cross = f"{rel}/src/scfm_pfm/data/crosscheck.py"
    univ = f"{rel}/src/scfm_pfm/genes/universe.py"
    demo = f"{rel}/src/scfm_pfm/models/demo.py"
    return {
        "filter_funnel": {
            "title": "Cells surviving each filter step, both cell lines",
            "description": (
                "Cells remaining after each filter in the order they are applied, from "
                "the raw deposited matrix to the final working set."
            ),
            "findings": (
                "Per-cell quality control removes almost nothing, because the deposited "
                f"matrices are already filtered: {ctx['mito_removed_hepg2']:,} HepG2 "
                "cells go to the mitochondrial-fraction cut and "
                f"{ctx['min_counts_removed_jurkat']:,} Jurkat cells to the minimum-count "
                "cut. The attrition that matters is at the perturbation level, where "
                f"requiring {C.PERTURBATION_FILTERS.min_cells} cells across "
                f"{C.PERTURBATION_FILTERS.min_batches} batches takes 2,393 HepG2 "
                f"perturbations down to {ctx['pert_hepg2']:,}."
            ),
            "sources": src("filter_funnel", prep),
        },
        "knockdown_distribution": {
            "title": "On-target knockdown per perturbation",
            "description": (
                "How much of the target gene's control-level expression each "
                "perturbation removed, as one histogram per cell line. The marked line "
                f"is the {C.EFFECTS.knockdown_detect_min:.2f} detection floor below "
                "which a perturbation is flagged rather than dropped."
            ),
            "findings": (
                f"The mass sits near complete knockdown (median "
                f"{_fmt(ctx['kd_hepg2'])} in HepG2, {_fmt(ctx['kd_jurkat'])} in Jurkat), "
                "so the CRISPRi perturbation did what it claims for most targets, and "
                f"the {ctx['kd_below_hepg2']:,} HepG2 and {ctx['kd_below_jurkat']:,} "
                "Jurkat perturbations below the floor are identifiable by name."
            ),
            "sources": src("knockdown_distribution", prep),
        },
        "knockdown_vs_effect": {
            "title": "On-target knockdown against transcriptome-wide effect size",
            "description": (
                "Each retained perturbation's knockdown of its own target against the "
                "L2 norm of its log2 fold-change vector over expressed genes, which is "
                "the size of its response across the rest of the transcriptome."
            ),
            "findings": (
                "The two are nearly unrelated: Spearman "
                f"{_fmt(ctx['kd_effect_spearman_hepg2'])} in HepG2 and "
                f"{_fmt(ctx['kd_effect_spearman_jurkat'])} in Jurkat. Silencing a gene "
                "efficiently does not make its downstream response large, so selecting "
                "perturbations by knockdown strength does not select the ones a model "
                "has a chance of predicting."
            ),
            "sources": src("knockdown_vs_effect", prep),
        },
        "effect_size_vs_de": {
            "title": "Effect size against differentially expressed gene count, by stratum",
            "description": (
                "Every retained perturbation, with effect size on the x-axis and its "
                "number of differentially expressed genes on a logarithmic y-axis, "
                "shifted by one so perturbations with none of them remain visible. "
                "Colour marks the equal-count effect-size quartile."
            ),
            "findings": (
                "The two measures of a perturbation's strength agree in ordering but "
                "not in scale: median differentially expressed genes rise from "
                f"{ctx['de_by_stratum_hepg2'][0]:.0f} in the lowest HepG2 quartile to "
                f"{ctx['de_by_stratum_hepg2'][3]:.0f} in the highest, while the effect "
                "norm varies continuously inside every quartile."
            ),
            "sources": src("effect_size_vs_de", prep),
        },
        "reference_crosscheck": {
            "title": "Agreement with the published reanalysis, per gene and per perturbation",
            "description": (
                "Left: a random sample of per-gene log2 fold changes, ours against the "
                "published reference release, with the identity line. Right: the "
                "distribution of per-perturbation agreement, ours against the reference "
                "and our two control estimators against each other."
            ),
            "findings": (
                f"Pooled Spearman against the reference is {_fmt(ctx['cc_spearman'])} "
                f"over {ctx['cc_entries']:,} shared entries, and our two estimators "
                f"agree with each other at median Pearson {_fmt(ctx['own_lfc'])}. The "
                "scatter is tight along the identity line rather than along a different "
                "slope, so the two pipelines agree on scale as well as on ordering."
            ),
            "sources": src("reference_crosscheck", cross),
        },
        "noise_ceilings": {
            "title": "Replicate agreement of a perturbation with itself, by stratum",
            "description": (
                "Median agreement between two independent halves of the same "
                "perturbation, for each effect-size quartile, each metric, each cell "
                "line, and both split forms. Solid bars split the cells; hatched bars "
                "split the sequencing batches, which forces disjoint control cells."
            ),
            "findings": (
                "Reproducibility rises steeply with effect size and stays low in the "
                f"bottom half: median agreement on log2 fold changes is "
                f"{_fmt(ctx['ceil_hepg2'][0])} in the lowest HepG2 quartile against "
                f"{_fmt(ctx['ceil_hepg2'][3])} in the highest. The two split forms give "
                "nearly the same answer, so the ceiling is not an artifact of sharing "
                "control cells."
            ),
            "sources": src("noise_ceilings", ceil),
        },
        "topk_overlap_grid": {
            "title": "Top-k overlap against the chance level it has to beat",
            "description": (
                "Median overlap between the two split halves' top-k gene sets, over "
                "k = 10, 50, 100 and 500 on a logarithmic axis, by effect-size stratum "
                "and cell line. The dashed line is the chance level k/n for n expressed "
                "genes; the open square marks the k quoted for that stratum and the "
                "cross the median adaptive k. The lower panels show the same numbers as "
                "excess over chance, where 0 is chance and 1 is an identical set."
            ),
            "findings": (
                "Raw overlap climbs with k in every stratum, and most of that climb is "
                "the chance level climbing underneath it: in the weakest HepG2 quartile "
                f"the top-500 overlap of {_fmt(ctx.get('grid_s1_500_hepg2', float('nan')))} "
                f"sits against a chance level of "
                f"{_sig(ctx.get('grid_chance_500_hepg2', float('nan')))}, while the "
                f"strongest quartile reaches {_fmt(ctx.get('grid_s4_500_hepg2', float('nan')))}. "
                "In the excess panels the curves are nearly flat, so the ranking of the "
                "strata, not the value of k, is what the metric is telling you."
            ),
            "sources": src("topk_overlap_grid", ceil),
        },
        "ceiling_vs_cells": {
            "title": "Replicate agreement against cells per perturbation",
            "description": (
                "The same split-half agreement, now per perturbation against how many "
                "cells that perturbation has, on a logarithmic x-axis, with the median "
                "of each cell-count bin drawn over the points."
            ),
            "findings": (
                "More cells help, but effect size decides: agreement correlates with "
                f"cell count at Spearman {_fmt(ctx['cells_vs_ceiling_hepg2'])} in HepG2 "
                f"and with effect size at {_fmt(ctx['effect_vs_ceiling_hepg2'])}. The "
                "weakest quartile is not the shallowest one, so its low agreement is a "
                "small-signal problem rather than an undersampling problem."
            ),
            "sources": src("ceiling_vs_cells", ceil),
        },
        "model_gene_coverage": {
            "title": "Vocabulary size and assay coverage for the four models",
            "description": (
                "For each model: how many tokens its vocabulary holds, how many resolve "
                "to a distinct Ensembl gene id, and how many of the genes measured in "
                "each cell line it can address. The axis is logarithmic because the "
                "vocabularies differ by a factor of three; the marked line is the shared "
                "universe."
            ),
            "findings": (
                f"All four models can address the shared {ctx['universe']:,} genes, but "
                "they get there from very different vocabularies, and scGPT's "
                f"{ctx['scgpt_unmapped']:,} unmapped vocabulary tokens are the reason "
                "the universe is defined by intersection rather than by any one model's "
                "gene list."
            ),
            "sources": src("model_gene_coverage", univ),
        },
        "model_responses_on_one_cell": {
            "title": "Each model's per-gene response on the same control cell",
            "description": (
                "One panel per model: the response the model returns for each of the "
                f"{ctx['universe']:,} universe genes, against that gene's observed count "
                "in the cell. Vertical scales are per panel and are not comparable."
            ),
            "findings": (
                "The four panels have different vertical ranges and different shapes "
                "because the models emit four different quantities, and each response "
                "tracks the observed counts only loosely (Spearman "
                f"{_fmt(ctx['resp_spearman_min'])} to {_fmt(ctx['resp_spearman_max'])}), "
                "so none of them is simply echoing its input."
            ),
            "sources": src("model_responses_on_one_cell", demo),
        },
    }


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------


def compose(root: Path, rel: str) -> tuple[list[Section], dict]:
    qc = {cl: json.loads((root / "effects" / cl / "qc.json").read_text()) for cl in LINES}
    pert = {
        cl: pd.read_parquet(root / "effects" / cl / "perturbations.parquet") for cl in LINES
    }
    ceil = {
        cl: pd.read_parquet(root / "effects" / cl / "noise_ceiling_summary.parquet")
        for cl in LINES
    }
    genes = {cl: pd.read_parquet(root / "effects" / cl / "genes.parquet") for cl in LINES}
    grid_files = {
        cl: root / "effects" / cl / "topk_overlap_grid_summary.parquet" for cl in LINES
    }
    has_grid = all(p.exists() for p in grid_files.values())
    grid = {cl: pd.read_parquet(p) for cl, p in grid_files.items()} if has_grid else {}
    grid_qc = (
        {
            cl: json.loads(
                (root / "effects" / cl / "topk_overlap_grid_qc.json").read_text()
            )
            for cl in LINES
        }
        if has_grid
        else {}
    )
    cov = json.loads((root / "gene_universe" / "coverage.json").read_text())
    cc_path = root / "crosscheck" / "summary.json"
    cc = json.loads(cc_path.read_text()) if cc_path.exists() else None

    contracts, demos, envs = {}, {}, {}
    for m in MODELS:
        for store, fname in ((contracts, "contract.json"), (demos, "demo.json"), (envs, "environment.json")):
            f = root / "models" / m / fname
            if f.exists():
                store[m] = json.loads(f.read_text())

    STRATA = list(C.STRATUM_NAMES)

    def ceil_med(cl: str, split: str, stratum: str, metric: str = "pearson_lfc") -> float:
        sub = ceil[cl][
            (ceil[cl].metric == metric)
            & (ceil[cl].split == split)
            & (ceil[cl].stratum == stratum)
        ]
        return float(sub["median"].iloc[0]) if not sub.empty else float("nan")

    def spear(cl: str, a: str, b: str) -> float:
        p = pert[cl]
        s = p[p[a].notna() & p[b].notna()]
        return float(s[a].corr(s[b], method="spearman"))

    def gcell(
        cl: str,
        stratum: str,
        k_label: str,
        column: str = "median_overlap",
        split: str = "split_half_cells",
    ) -> float:
        sub = grid[cl][
            (grid[cl].split == split)
            & (grid[cl].k_label == k_label)
            & (grid[cl].stratum == stratum)
        ]
        return float(sub[column].iloc[0]) if not sub.empty else float("nan")

    ctx = {
        "universe": cov["universe_size"],
        "pert_hepg2": qc["hepg2"]["retained"]["perturbations"],
        "pert_jurkat": qc["jurkat"]["retained"]["perturbations"],
        "kd_hepg2": qc["hepg2"]["knockdown"]["median"],
        "kd_jurkat": qc["jurkat"]["knockdown"]["median"],
        "kd_below_hepg2": qc["hepg2"]["knockdown"]["n_below_detect_threshold"],
        "kd_below_jurkat": qc["jurkat"]["knockdown"]["n_below_detect_threshold"],
        "mito_removed_hepg2": qc["hepg2"]["filter_funnel"][4]["removed"],
        "min_counts_removed_jurkat": qc["jurkat"]["filter_funnel"][2]["removed"],
        "kd_effect_spearman_hepg2": spear("hepg2", "knockdown", "effect_l2_lfc"),
        "kd_effect_spearman_jurkat": spear("jurkat", "knockdown", "effect_l2_lfc"),
        "cells_vs_ceiling_hepg2": spear("hepg2", "n_cells", "split_half_cells__pearson_lfc"),
        "cells_vs_ceiling_jurkat": spear("jurkat", "n_cells", "split_half_cells__pearson_lfc"),
        "effect_vs_ceiling_hepg2": spear(
            "hepg2", "effect_l2_lfc", "split_half_cells__pearson_lfc"
        ),
        "effect_vs_ceiling_jurkat": spear(
            "jurkat", "effect_l2_lfc", "split_half_cells__pearson_lfc"
        ),
        "cells_vs_effect_hepg2": spear("hepg2", "n_cells", "effect_l2_lfc"),
        "cells_vs_effect_jurkat": spear("jurkat", "n_cells", "effect_l2_lfc"),
        "ceil_hepg2": [ceil_med("hepg2", "split_half_cells", s) for s in STRATA],
        "ceil_jurkat": [ceil_med("jurkat", "split_half_cells", s) for s in STRATA],
        "de_by_stratum_hepg2": [
            float(pert["hepg2"][pert["hepg2"].effect_stratum == s].n_de_genes.median())
            for s in STRATA
        ],
        "scgpt_unmapped": cov["coverage"]["scgpt"]["n_unmapped_tokens"],
        "median_cells_per_batch_hepg2": float(
            (pert["hepg2"].n_cells / pert["hepg2"].n_batches).median()
        ),
        "grid_repro_max": max(
            (
                max(q["reproduction_of_published_ceilings"]["max_abs_diff"].values())
                for q in grid_qc.values()
            ),
            default=float("nan"),
        ),
        "grid_chance_500_hepg2": gcell("hepg2", "all", "500", "chance") if has_grid else float("nan"),
        "grid_s1_500_hepg2": gcell("hepg2", STRATA[0], "500") if has_grid else float("nan"),
        "grid_s4_500_hepg2": gcell("hepg2", STRATA[-1], "500") if has_grid else float("nan"),
        "n_expressed_hepg2": int(genes["hepg2"].expressed.sum()),
        "n_expressed_jurkat": int(genes["jurkat"].expressed.sum()),
        "de_by_stratum_jurkat": [
            float(pert["jurkat"][pert["jurkat"].effect_stratum == s].n_de_genes.median())
            for s in STRATA
        ],
        "cc_spearman": cc["pooled_spearman"] if cc else float("nan"),
        "cc_entries": cc["n_pooled_entries"] if cc else 0,
        "own_lfc": cc["own_estimator_agreement"]["median_pearson_within_vs_pooled_lfc"]
        if cc
        else float("nan"),
    }
    resp_sp = [
        demos[m]["response_summary"]["spearman_response_vs_input"] for m in MODELS if m in demos
    ]
    ctx["resp_spearman_min"] = min(resp_sp) if resp_sp else float("nan")
    ctx["resp_spearman_max"] = max(resp_sp) if resp_sp else float("nan")

    prep_src = (
        f"{rel}/src/scfm_pfm/data/prepare.py;{rel}/src/scfm_pfm/config.py;"
        f"{rel}/src/scfm_pfm/data/sources.py"
    )
    ceil_src = f"{rel}/src/scfm_pfm/data/ceilings.py;{rel}/src/scfm_pfm/data/prepare.py"
    cross_src = (
        f"{rel}/src/scfm_pfm/data/crosscheck.py;{rel}/src/scfm_pfm/data/crosscheck_input.py"
    )
    model_src = (
        f"{rel}/src/scfm_pfm/models/demo.py;{rel}/src/scfm_pfm/models/base.py;"
        f"{rel}/src/scfm_pfm/genes/universe.py"
    )

    S: list[Section] = []

    # ---------------- opening ----------------
    opening = Section(sources=f"{prep_src};{model_src}".rsplit(";", 1)[0])
    opening.blocks += [
        Head(1, TITLE),
        Para(
            "This run measured perturbation effects for two CRISPRi Perturb-seq cell "
            f"lines and wrote down, with measurements rather than assertions, what four "
            f"single-cell foundation models emit when asked about a cell. In HepG2 "
            f"{ctx['pert_hepg2']:,} perturbations and in Jurkat {ctx['pert_jurkat']:,} "
            "survive filtering, their fold changes agree with an independent published "
            f"reanalysis at pooled Spearman {_fmt(ctx['cc_spearman'])} over "
            f"{ctx['cc_entries']:,} gene-level entries, and each of the four models now "
            f"has a per-gene response vector and a Jacobian column over a shared "
            f"{ctx['universe']:,}-gene universe on one real control cell."
        ),
        Para(
            "The most consequential number here is not an agreement statistic but a "
            "limit. Splitting each perturbation's cells in two and comparing the two "
            "halves against each other, the weaker half of the perturbations barely "
            f"reproduce themselves: median agreement is {_fmt(ctx['ceil_hepg2'][0])} and "
            f"{_fmt(ctx['ceil_hepg2'][1])} in the two lowest HepG2 quartiles against "
            f"{_fmt(ctx['ceil_hepg2'][3])} in the highest. Any later comparison of model "
            "predictions has to be read against that limit rather than against a perfect "
            "score."
        ),
        Para(
            "Each of the three areas below is written in two parts: first the quantity "
            "and the exact procedure that produced it, including sample sizes and "
            "thresholds, then the measured values with what each one implies. Every "
            "number is read at build time out of the artifact files the run produced, "
            "which are listed at the end."
        ),
        Para(
            f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._"
        ),
    ]
    S.append(opening)

    # ---------------- effect matrices ----------------
    em = Section(sources=prep_src)
    em.blocks += [
        Head(2, "The effect matrices"),
        Head(3, "Definition, procedure, and thresholds"),
        Para(
            "An effect matrix is one table per cell line. **A row is one CRISPRi "
            "perturbation**: a single target gene, with every cell carrying any guide "
            "against that target pooled together, so the unit is the target gene and "
            "not the individual guide. **A column is one gene in the assay's measured "
            "panel.** The entry at a row and column is how much that gene's expression "
            "changed in cells carrying that perturbation, relative to non-targeting "
            "control cells."
        ),
        Para(
            "Each cell is first normalized to counts per 10,000 over the measured panel "
            "(CP10K), and a perturbation's expression profile is the mean of those "
            "normalized per-cell vectors, not a normalized sum of raw counts. The change "
            "is reported in two forms: `delta`, the difference of mean CP10K, and "
            "`log2fc`, the base-2 logarithm of their ratio with a small constant "
            f"({C.EFFECTS.lfc_eps:g} CP10K) added to both sides so a zero control level "
            "cannot produce an infinite value."
        ),
        Para(
            "Two control estimators are computed for each form, which is why "
            "`effects.npz` holds four matrices. **The batch-matched estimator is the "
            "primary one** (`delta_within`, `lfc_within`). It computes the mean control "
            "profile separately in each sequencing batch (`gem_group`), then builds the "
            "reference for a perturbation as the average of those per-batch control "
            "means weighted by the share of that perturbation's own cells in each batch. "
            "The batch-pooled estimator (`delta_pooled`, `lfc_pooled`) references every "
            "perturbation against one control mean over all control cells, and is kept "
            "as a secondary column for comparison. Row order follows "
            "`perturbations.parquet` and column order follows `genes.parquet`; the "
            "matrices carry no labels of their own, so the indices mean nothing without "
            "those two tables."
        ),
        Para(
            "The batch-matched reference is weighted rather than computed strictly "
            "within batch because these libraries are thin: the median perturbation here "
            f"has {ctx['median_cells_per_batch_hepg2']:.1f} cells per batch it appears in "
            "(HepG2). A strictly within-batch contrast would estimate each batch's effect "
            "from one or two perturbed cells, drop every batch that happens to hold no "
            "control cell, and then average those noisy pieces. Weighting the control "
            "side by the perturbation's batch composition removes the same first-order "
            "confound, which is that a perturbation over-represented in one batch would "
            "otherwise inherit that batch's baseline, while keeping all of the "
            "perturbation's cells on the numerator side. The two estimators are compared "
            "directly in the cross-check below, so the size of the correction is a "
            "measured number rather than an assumption."
        ),
        Para(
            "Differential expression is scored differently on purpose. For each gene, a "
            "**Welch t-test on log1p(CP10K)** compares that perturbation's cells against "
            "**pooled** non-targeting controls, and Benjamini-Hochberg control at "
            f"{C.EFFECTS.de_fdr:.2f} is applied across expressed genes; `n_de_genes` "
            "counts the genes that pass. The test uses pooled controls because a t-test "
            "needs a variance and a sample size on both sides, and the batch-matched "
            "reference is a weighted mean of per-batch means, a point estimate with no "
            "per-cell spread attached; the individual batches hold far too few control "
            "cells to estimate a per-gene variance. Pooling all "
            f"{qc['hepg2']['retained']['control_cells']:,} HepG2 and "
            f"{qc['jurkat']['retained']['control_cells']:,} Jurkat control cells gives "
            "the most stable variance and the largest degrees of freedom, and batch "
            "composition mainly shifts the control mean, which the effect estimate "
            "already corrects, rather than inflating the within-gene variance. The "
            "consequence to keep in mind is that the differential-expression count and "
            "the reported fold change are computed against slightly different references, "
            "so they are two related screens rather than two views of one number."
        ),
        Para(
            "A gene counts as **expressed** when its pooled control level is at least "
            f"{C.EFFECTS.expressed_min_cp10k:g} CP10K; effect sizes, the "
            "differential-expression correction, and every ceiling are restricted to "
            "those genes. **On-target knockdown** is one minus the ratio of the "
            "perturbed to the batch-matched control level at the target gene's own "
            f"column, and a perturbation whose knockdown falls below "
            f"{C.EFFECTS.knockdown_detect_min:.2f} is flagged rather than dropped, as is "
            "one whose target gene is not in the measured panel at all."
        ),
        Bullets(
            [
                "Cell filters: at least "
                f"{C.CELL_FILTERS.min_counts:,} UMI, at least "
                f"{C.CELL_FILTERS.min_genes_detected} genes detected, mitochondrial "
                f"fraction at most {C.CELL_FILTERS.max_mito_fraction:g}.",
                "Perturbation filters: at least "
                f"{C.PERTURBATION_FILTERS.min_cells} quality-passing cells spread over "
                f"at least {C.PERTURBATION_FILTERS.min_batches} sequencing batches.",
                f"Expressed genes: pooled control level at least "
                f"{C.EFFECTS.expressed_min_cp10k:g} CP10K.",
                f"Differential expression: Benjamini-Hochberg FDR below "
                f"{C.EFFECTS.de_fdr:.2f} across expressed genes.",
                f"Seed {C.SEED} for every split and sample.",
            ]
        ),
        Para(
            f"**The effect-size strata are equal-count quartiles, cut per cell line.** "
            "All retained perturbations in a cell line are ranked by `effect_l2_lfc`, "
            "the Euclidean norm of their batch-matched log2 fold-change vector over "
            "expressed genes, with ties broken by first occurrence. The ranks are then "
            f"divided into {C.EFFECTS.n_strata} groups of equal size: the lowest quarter "
            f"of ranks becomes `{STRATA[0]}` and the highest becomes `{STRATA[-1]}`. "
            "Because the cut is on ranks within a cell line, the stratum boundaries are "
            "not fixed effect-size values and are not the same in the two cell lines; "
            "the realized ranges are in the table below, and a stratum label should "
            "never be read as an absolute effect size across cell lines."
        ),
        FigRef("filter_funnel"),
        Head(3, "Results and what they imply"),
    ]

    main_rows = []

    def row(name, fn):
        main_rows.append([name, fn("hepg2"), fn("jurkat")])

    row("cells × measured genes", lambda cl: f"{qc[cl]['shape']['cells']:,} × {qc[cl]['shape']['genes']:,}")
    row("source file size", lambda cl: f"{qc[cl]['source']['size_bytes'] / 1e9:.3f} GB")
    row("sequencing batches", lambda cl: f"{qc[cl]['n_gem_groups']}")
    row("cells after quality control", lambda cl: f"{qc[cl]['retained']['cells']:,}")
    row("non-targeting control cells", lambda cl: f"{qc[cl]['retained']['control_cells']:,}")
    row("retained perturbations", lambda cl: f"{qc[cl]['retained']['perturbations']:,}")
    row(
        f"expressed genes (CP10K ≥ {C.EFFECTS.expressed_min_cp10k:g})",
        lambda cl: f"{int(genes[cl].expressed.sum()):,} of {genes[cl].shape[0]:,}",
    )
    row("median on-target knockdown", lambda cl: _fmt(qc[cl]["knockdown"]["median"]))
    row(
        "perturbations below the knockdown floor",
        lambda cl: f"{qc[cl]['knockdown']['n_below_detect_threshold']:,}",
    )
    row(
        "target gene not in the measured panel",
        lambda cl: f"{qc[cl]['knockdown']['n_target_gene_not_measured']:,}",
    )
    row("median differentially expressed genes", lambda cl: f"{int(pert[cl].n_de_genes.median()):,}")
    row("median cells per perturbation", lambda cl: f"{int(pert[cl].n_cells.median()):,}")
    row("processing runtime", lambda cl: f"{qc[cl]['runtime_seconds']:.0f} s")
    em.blocks += [
        Table(["", "HepG2", "Jurkat"], main_rows),
        Para(
            "Both cell lines pass the checks a Perturb-seq effect matrix has to pass "
            "before it is worth modelling: knockdown is nearly complete at the median, "
            "the retained perturbations each have tens of cells across at least two "
            "batches, and a third of the measured panel is expressed highly enough to "
            "carry a fold change. Jurkat is the deeper of the two datasets in cells but "
            "the noisier one per perturbation, which shows up in the ceilings below."
        ),
        FigRef("knockdown_distribution"),
        FigRef("knockdown_vs_effect"),
        Para(
            "That has a practical consequence for how a later run picks its evaluation "
            "set. Perturbations with a large measurable response cannot be selected by "
            "knockdown strength; they have to be selected by effect size or by stratum, "
            "which is what the strata below exist for. Knockdown remains the check that "
            "the intended perturbation happened, not a proxy for how much it did."
        ),
    ]

    strata_rows = []
    for i, s in enumerate(STRATA):
        cells = []
        for cl in LINES:
            p = pert[cl]
            sub = p[p.effect_stratum == s]
            cells += [
                f"{sub.shape[0]:,}",
                f"{_fmt(sub.effect_l2_lfc.min(), 2)} – {_fmt(sub.effect_l2_lfc.max(), 2)}",
                f"{sub.n_de_genes.median():.0f}",
                f"{int(sub.n_cells.median()):,}",
            ]
        strata_rows.append([s] + cells)
    em.blocks += [
        Table(
            [
                "stratum",
                "HepG2 n",
                "HepG2 effect-size range",
                "HepG2 median DE genes",
                "HepG2 median cells",
                "Jurkat n",
                "Jurkat effect-size range",
                "Jurkat median DE genes",
                "Jurkat median cells",
            ],
            strata_rows,
        ),
        Para(
            "The realized boundaries show why the strata are worth carrying: the effect "
            "norm spans roughly an order of magnitude from the lowest to the highest "
            "quartile, and the number of differentially expressed genes spans two. They "
            "also show the trap in the labels, since the same label covers different "
            "absolute ranges in the two cell lines, so `S4_highest` in Jurkat is not the "
            "same experimental object as `S4_highest` in HepG2."
        ),
        FigRef("effect_size_vs_de"),
    ]
    S.append(em)

    # ---------------- ceilings ----------------
    nc = Section(sources=ceil_src)
    nc.blocks += [
        Head(2, "The noise ceilings"),
        Head(3, "Definition, procedure, and thresholds"),
        Para(
            "A noise ceiling answers a question about the data, not about any model: "
            "**if the same perturbation were measured twice at this sequencing depth, "
            "how similar would the two measurements be?** Each perturbation's "
            "quality-passing cells are divided into two sides, the effect vector is "
            "computed independently on each side using exactly the pipeline described "
            "above, and the two vectors are compared across expressed genes. Whatever "
            "agreement they reach is the reproducible part of the measurement; the rest "
            "is sampling noise that no predictor can recover, because it is not a "
            "property of the perturbation at all."
        ),
        Para(
            "Two split forms are reported, and both keep the control cells of the two "
            "sides disjoint so shared control noise cannot inflate the agreement. "
            "`split_half_cells` splits the perturbation's cells in half by a seeded "
            "assignment and references each half against its own half of the "
            "non-targeting control cells. `split_by_batch` alternates the perturbation's "
            "sequencing batches into two sets, largest batch first, and references each "
            "set against the control cells of its own batches, so disjointness holds by "
            "construction rather than by the seed. A side with fewer than "
            f"{ceil_min_cells()} cells is not scored, which is why some perturbations "
            "carry a ceiling for one split form and not the other."
        ),
        Para(
            "Three metrics are computed per split, chosen to match the ways a later run "
            "will score a model against these effects: Pearson correlation across "
            "expressed genes on log2 fold changes (`pearson_lfc`), cosine similarity on "
            "the linear CP10K deltas (`cosine_delta`), and the overlap of the top "
            f"{C.EFFECTS.topk} genes ranked by absolute log2 fold change "
            "(`topk_overlap`). Per-perturbation values live in "
            "`perturbations.parquet`; `noise_ceiling_summary.parquet` holds the median "
            "and the quartiles per stratum."
        ),
        *(
            [
                Para(
                    "The overlap metric is reported over a grid of k rather than at one "
                    "k. For every perturbation and both split forms, the two sides' "
                    "top-k sets are compared at k = "
                    + ", ".join(str(k) for k in ceilings_grid.K_GRID)
                    + ", and at an adaptive k defined below. The expensive part of this "
                    "measurement is rebuilding the two split-half effect vectors from "
                    "the raw count matrices; once they exist, another k costs "
                    "microseconds, so the whole grid is computed and any k can be read "
                    "off it. The k = "
                    f"{C.EFFECTS.topk} column is the `topk_overlap` column already in "
                    "`perturbations.parquet`, recomputed and identical to it. "
                    "Per-perturbation values live in `topk_overlap_grid.parquet`, "
                    "per-stratum medians in `topk_overlap_grid_summary.parquet`, and "
                    "the split-half vectors themselves in `split_half_vectors.npz` "
                    "(log2FC and delta, both sides, both split forms, float32 over "
                    "expressed genes), so a later question about another k or another "
                    "metric is a file read rather than another pass over the raw data. "
                    "The stage re-reads the raw count matrices, because those vectors "
                    "were not kept the first time, and it reproduces every published "
                    "ceiling column exactly on the way through (largest absolute "
                    f"difference {ctx['grid_repro_max']:g} over both split forms and all "
                    "three metrics), so nothing already reported changes meaning."
                ),
                Para(
                    "**Overlap has a floor that moves with k, so raw overlap is not "
                    "comparable across k.** Two independent sets of size k drawn from n "
                    "expressed genes share k/n of their members by chance. In HepG2, "
                    f"with {ctx['n_expressed_hepg2']:,} expressed genes, that chance "
                    f"level is {_sig(gcell('hepg2', 'all', '10', 'chance'))} at k = 10 "
                    f"and {_sig(gcell('hepg2', 'all', '500', 'chance'))} at k = 500; in "
                    f"Jurkat, with {ctx['n_expressed_jurkat']:,} expressed genes, "
                    f"{_sig(gcell('jurkat', 'all', '10', 'chance'))} and "
                    f"{_sig(gcell('jurkat', 'all', '500', 'chance'))}. A top-500 overlap "
                    "of 0.30 is therefore under twice what two random draws reach, while "
                    "a top-10 overlap of 0.30 is roughly a hundred times chance. "
                    "Every cell of the grid is reported twice for that reason: the raw "
                    "overlap, and the excess over chance, (overlap - chance) / "
                    "(1 - chance), which is 0 when the two halves agree no better than "
                    "two random draws and 1 when they select exactly the same genes."
                ),
                Para(
                    "A fifth column sets k per perturbation rather than per stratum: k "
                    "is that perturbation's own `n_de_genes`, clipped to "
                    f"[{ceilings_grid.ADAPTIVE_K_MIN}, {ceilings_grid.ADAPTIVE_K_MAX}]. "
                    "The strata are rank quartiles cut inside a cell line, so a stratum "
                    "label does not fix a response size: the median S3 perturbation has "
                    f"{ctx['de_by_stratum_hepg2'][2]:.0f} differentially expressed genes "
                    f"in HepG2 against {ctx['de_by_stratum_jurkat'][2]:.0f} in Jurkat, "
                    f"and the median S4 perturbation {ctx['de_by_stratum_hepg2'][3]:.0f} "
                    f"against {ctx['de_by_stratum_jurkat'][3]:.0f}. Asking both lines for "
                    "their top 500 genes asks a much harder question of Jurkat's top "
                    "stratum than of HepG2's. A k tied to the perturbation's own "
                    "response asks both the same question, at the cost of a chance level "
                    "that changes from row to row, so the adaptive column carries its "
                    "own chance value per perturbation."
                ),
                Para(
                    "Two diagnostics travel with every cell, because the overlap can be "
                    "quietly meaningless without them. Selecting the top k with "
                    "`argpartition` breaks ties arbitrarily, which matters at small k "
                    "where many genes sit near zero log2 fold change, so `tie_fraction` "
                    "records the share of the selected set within numerical tolerance "
                    f"(relative {ceilings_grid.TIE_RTOL:g}) of the k-th value; its floor "
                    "is 1/k because the k-th value ties itself, and membership is only "
                    "actually contested when a gene outside the set falls inside that "
                    "tolerance too. Selection is on absolute log2 fold change with no "
                    "expression or significance filter, so a small-k set can be "
                    "dominated by low-expression genes whose ratios are large and "
                    "unreliable; `median_control_cp10k_selected` records the median "
                    "pooled control level of the selected genes, to be read against the "
                    "median over all expressed genes ("
                    f"{_fmt(float(genes['hepg2'][genes['hepg2'].expressed].pooled_control_cp10k.median()))} "
                    "CP10K in HepG2, "
                    f"{_fmt(float(genes['jurkat'][genes['jurkat'].expressed].pooled_control_cp10k.median()))} "
                    "in Jurkat)."
                ),
            ]
            if has_grid
            else []
        ),
        Para(
            "**A model score has to be read as a fraction of this number, not on its "
            "own.** If two independent measurements of the same perturbation agree at "
            "0.30, then a prediction that correlates with a measured effect vector at "
            "0.30 has reached the level at which the experiment agrees with itself, and "
            "a prediction at 0.30 in a stratum whose ceiling is 0.70 has reached less "
            "than half of it. Reporting the raw correlation alone makes the first look "
            "as weak as the second. The ceilings are per stratum and per cell line for "
            "exactly this reason: they are the denominators."
        ),
        Head(3, "Results and what they imply"),
    ]

    ceil_rows = []
    for s in STRATA:
        ceil_rows.append(
            [
                s,
                _fmt(ceil_med("hepg2", "split_half_cells", s)),
                _fmt(ceil_med("hepg2", "split_by_batch", s)),
                _fmt(ceil_med("hepg2", "split_half_cells", s, "cosine_delta")),
                _fmt(ceil_med("jurkat", "split_half_cells", s)),
                _fmt(ceil_med("jurkat", "split_by_batch", s)),
                _fmt(ceil_med("jurkat", "split_half_cells", s, "cosine_delta")),
            ]
        )
    n_bottom_hepg2 = int(
        pert["hepg2"].effect_stratum.isin(STRATA[:2]).sum()
    )
    n_bottom_jurkat = int(pert["jurkat"].effect_stratum.isin(STRATA[:2]).sum())
    nc.blocks += [
        Table(
            [
                "stratum",
                "HepG2 split by cell",
                "HepG2 split by batch",
                "HepG2 cosine on delta",
                "Jurkat split by cell",
                "Jurkat split by batch",
                "Jurkat cosine on delta",
            ],
            ceil_rows,
        ),
        Para(
            "Agreement rises with effect size in both cell lines, in both split forms, "
            "and in both metrics, and the two split forms land on nearly the same value "
            "in every cell of the table. That is what a working ceiling estimate looks "
            "like, and it means the ordering is a property of the perturbations rather "
            "than of the way the halves were drawn."
        ),
        Para(
            "**The bottom two strata are effectively unpredictable at this sequencing "
            f"depth.** Their log2 fold-change vectors reproduce at "
            f"{_fmt(ctx['ceil_hepg2'][0])} and {_fmt(ctx['ceil_hepg2'][1])} in HepG2 and "
            f"{_fmt(ctx['ceil_jurkat'][0])} and {_fmt(ctx['ceil_jurkat'][1])} in Jurkat, "
            "which is close enough to zero that the measured effect vector is mostly "
            f"noise. That covers {n_bottom_hepg2:,} of the {ctx['pert_hepg2']:,} HepG2 "
            f"perturbations and {n_bottom_jurkat:,} of the {ctx['pert_jurkat']:,} Jurkat "
            "perturbations, about half of each dataset. Two consequences follow for "
            "everything built on these matrices. A model that scores near zero there has "
            "not been shown to fail, and a model that scores slightly above another one "
            "has not been shown to be better; the measurement cannot separate them. Any "
            "later claim about model quality on this data therefore has to be restricted "
            "to the upper strata, or stated as a fraction of the stratum's ceiling, and "
            "a headline number pooled over all four strata will be dominated by "
            "perturbations whose truth is unmeasured."
        ),
        Para(
            "Cosine similarity on the linear deltas stays high everywhere "
            f"({_fmt(ceil_med('hepg2', 'split_half_cells', STRATA[-1], 'cosine_delta'))} "
            "in the top HepG2 stratum) because it is dominated by the few "
            "highly-expressed genes with large absolute changes. It is the easier metric "
            "and should not be quoted as though it were the correlation."
        ),
        FigRef("noise_ceilings"),
        FigRef("ceiling_vs_cells"),
        Para(
            "The low ceiling in the weak strata is not a matter of too few cells. "
            "Agreement correlates with effect size at Spearman "
            f"{_fmt(ctx['effect_vs_ceiling_hepg2'])} (HepG2) and "
            f"{_fmt(ctx['effect_vs_ceiling_jurkat'])} (Jurkat) but with cell count at "
            f"only {_fmt(ctx['cells_vs_ceiling_hepg2'])} and "
            f"{_fmt(ctx['cells_vs_ceiling_jurkat'])}, and cell count runs *against* "
            f"effect size ({_fmt(ctx['cells_vs_effect_hepg2'])} in HepG2, "
            f"{_fmt(ctx['cells_vs_effect_jurkat'])} in Jurkat): the weakest quartile has "
            f"a median of {int(pert['hepg2'][pert['hepg2'].effect_stratum == STRATA[0]].n_cells.median()):,} "
            "cells in HepG2 against "
            f"{int(pert['hepg2'][pert['hepg2'].effect_stratum == STRATA[-1]].n_cells.median()):,} "
            "in the strongest. The weak perturbations were measured more deeply and "
            "still did not reproduce, so their response is small relative to cell-level "
            "variation rather than merely undersampled, and sequencing more cells of the "
            "same perturbations is not an obvious route to a usable signal there."
        ),
    ]

    if has_grid:
        k_labels = [str(k) for k in ceilings_grid.K_GRID]
        grid_header = (
            ["cell line", "stratum"]
            + [f"k = {k}" for k in k_labels]
            + ["adaptive k"]
        )
        grid_rows = []
        for cl in LINES:
            grid_rows.append(
                [LABEL[cl], "chance level"]
                + [_sig(gcell(cl, "all", lab, "chance")) for lab in k_labels]
                + [f"{_sig(gcell(cl, 'all', 'adaptive', 'chance'))} (median)"]
            )
            for s in STRATA:
                cells = []
                for lab in k_labels:
                    cell = (
                        f"{_fmt(gcell(cl, s, lab))} "
                        f"({_fmt(gcell(cl, s, lab, 'median_excess_over_chance'))})"
                    )
                    if ceilings_grid.DIAGONAL_K.get(s) == int(lab):
                        cell = f"**{cell}**"
                    cells.append(cell)
                cells.append(
                    f"{_fmt(gcell(cl, s, 'adaptive'))} "
                    f"({_fmt(gcell(cl, s, 'adaptive', 'median_excess_over_chance'))})"
                    f" at k = {gcell(cl, s, 'adaptive', 'median_k'):g}"
                )
                grid_rows.append([LABEL[cl], s] + cells)

        diag_header = [
            "cell line",
            "k",
            "median tie fraction",
            "selections with a contested boundary",
            "median control level of the selected genes (CP10K)",
        ]
        diag_rows = [
            [
                LABEL[cl],
                lab,
                _sig(gcell(cl, "all", lab, "median_tie_fraction")),
                f"{100 * gcell(cl, 'all', lab, 'frac_selection_contested'):.2f}%",
                _fmt(gcell(cl, "all", lab, "median_control_cp10k_selected")),
            ]
            for cl in LINES
            for lab in k_labels + ["adaptive"]
        ]

        def split_gap(cl: str) -> float:
            return max(
                abs(gcell(cl, s, lab) - gcell(cl, s, lab, split="split_by_batch"))
                for s in STRATA
                for lab in k_labels + ["adaptive"]
            )

        nc.blocks += [
            Head(4, "The top-k overlap grid"),
            Para(
                "Median overlap of the two halves' top-k gene sets, split by cell, with "
                "excess over chance in parentheses and each stratum's own k in bold. "
                "The chance row is the overlap two independent draws of the same size "
                "reach, and the adaptive column uses each perturbation's own "
                "differentially expressed gene count as its k. Splitting by sequencing "
                "batch instead of by cell moves no cell of this table by more than "
                f"{_fmt(max(split_gap(cl) for cl in LINES))}."
            ),
            Table(grid_header, grid_rows),
            Para(
                "**Reading across a row shows why the chance level has to be quoted "
                "with the number.** The weakest HepG2 quartile appears to improve from "
                f"{_fmt(gcell('hepg2', STRATA[0], '10'))} at k = 10 to "
                f"{_fmt(gcell('hepg2', STRATA[0], '500'))} at k = 500, which looks like "
                "a better measurement at larger k. It is not: chance rises from "
                f"{_sig(gcell('hepg2', 'all', '10', 'chance'))} to "
                f"{_sig(gcell('hepg2', 'all', '500', 'chance'))} over the same span, so "
                "the excess over chance stays near "
                f"{_fmt(gcell('hepg2', STRATA[0], '500', 'median_excess_over_chance'))} "
                "throughout. Two thirds of that top-500 overlap is what two random draws "
                "would have produced. The same row in Jurkat reads "
                f"{_fmt(gcell('jurkat', STRATA[0], '500'))} against a chance level of "
                f"{_sig(gcell('jurkat', 'all', '500', 'chance'))}."
            ),
            Para(
                "**Reading down a column shows which k can see the effect-size "
                "ordering.** At k = 50 and above, the top stratum separates cleanly from "
                "the bottom one in both cell lines: in HepG2 the excess runs "
                f"{_fmt(gcell('hepg2', STRATA[0], '100', 'median_excess_over_chance'))} "
                f"to {_fmt(gcell('hepg2', STRATA[-1], '100', 'median_excess_over_chance'))} "
                "at k = 100, and in Jurkat "
                f"{_fmt(gcell('jurkat', STRATA[0], '100', 'median_excess_over_chance'))} "
                f"to {_fmt(gcell('jurkat', STRATA[-1], '100', 'median_excess_over_chance'))}. "
                "At k = 10 it cannot: a per-perturbation overlap of ten genes takes only "
                "the eleven values 0, 0.1, ..., 1, so the median collapses onto that "
                "grid and three of the four HepG2 strata land on exactly "
                f"{_fmt(gcell('hepg2', STRATA[-1], '10'))}. k = 10 is a legitimate "
                "question about the few strongest genes, but it is too coarse to rank "
                "perturbations by, and a model compared at k = 10 on a single "
                "perturbation is being scored on a ten-way coin toss."
            ),
            FigRef("topk_overlap_grid"),
            Para(
                "**The adaptive column is the one that compares the two cell lines.** "
                "Each perturbation is asked for as many genes as it actually moved, so "
                "the median k is "
                f"{gcell('hepg2', STRATA[2], 'adaptive', 'median_k'):g} in HepG2's S3 "
                f"against {gcell('jurkat', STRATA[2], 'adaptive', 'median_k'):g} in "
                "Jurkat's, and "
                f"{gcell('hepg2', STRATA[-1], 'adaptive', 'median_k'):g} against "
                f"{gcell('jurkat', STRATA[-1], 'adaptive', 'median_k'):g} in S4. On "
                "that footing the top HepG2 stratum reproduces its own response at an "
                "excess of "
                f"{_fmt(gcell('hepg2', STRATA[-1], 'adaptive', 'median_excess_over_chance'))} "
                "against "
                f"{_fmt(gcell('jurkat', STRATA[-1], 'adaptive', 'median_excess_over_chance'))} "
                "in Jurkat, so the S4 label covers a substantially more reproducible set "
                "of perturbations in HepG2 than in Jurkat. In the bottom two strata the "
                "adaptive k collapses to the floor of "
                f"{ceilings_grid.ADAPTIVE_K_MIN} "
                "in HepG2, because the median perturbation there has fewer than ten "
                "differentially expressed genes at all, which is another way of saying "
                "the same thing the correlation ceiling says: there is little there to "
                "reproduce."
            ),
            Table(diag_header, diag_rows),
            Para(
                "The two diagnostics land differently. Ties are not a practical problem "
                "in this data: the median tie fraction is exactly the 1/k floor in every "
                "cell, so the only value tied with the k-th value is the k-th value "
                "itself, and the share of selections where any gene outside the set is "
                "indistinguishable from the boundary peaks at "
                f"{100 * max(gcell(cl, s, lab, 'frac_selection_contested') for cl in LINES for s in STRATA + ['all'] for lab in k_labels + ['adaptive']):.2f}% "
                "across every cell of the grid. The expression diagnostic is the one "
                "worth carrying: the selected genes sit below the typical expressed gene "
                "everywhere, from "
                f"{_fmt(min(gcell(cl, s, lab, 'median_control_cp10k_selected') for cl in LINES for s in STRATA + ['all'] for lab in k_labels + ['adaptive']))} "
                "to "
                f"{_fmt(max(gcell(cl, s, lab, 'median_control_cp10k_selected') for cl in LINES for s in STRATA + ['all'] for lab in k_labels + ['adaptive']))} "
                "CP10K against about 1 CP10K over all expressed genes, and the lean is "
                "strongest exactly where the signal is weakest "
                f"({_fmt(gcell('jurkat', STRATA[0], '10', 'median_control_cp10k_selected'))} "
                "CP10K for the lowest Jurkat stratum at k = 10 against "
                f"{_fmt(gcell('hepg2', STRATA[-1], '500', 'median_control_cp10k_selected'))} "
                "for the highest HepG2 stratum at k = 500). Ranking by absolute log2 "
                "fold change with no expression filter does pull in genes whose ratios "
                "are large because their denominators are small, so a later run that "
                "wants a top-k comparison to be about biology rather than about counting "
                "noise should either add an expression floor or read this column as the "
                "caveat it is."
            ),
            Para(
                "None of this changes a number that was already reported: the k = "
                f"{C.EFFECTS.topk} column is the published `topk_overlap` column, "
                "recomputed to the last bit, and the correlation and cosine ceilings are "
                "untouched. What the grid adds is the ability to quote an overlap "
                "honestly. A top-k overlap without its k, its chance level and its "
                "expressed-gene count is not interpretable, and the same overlap value "
                "can mean a strong result at k = 10 and a near-chance result at k = 500."
            ),
        ]
    S.append(nc)

    # ---------------- crosscheck ----------------
    if cc:
        xc = Section(sources=cross_src)
        xc.blocks += [
            Head(2, "Agreement with an independent reanalysis"),
            Head(3, "Definition, procedure, and thresholds"),
            Para(
                f"The reference release `{cc['reference']['repo']}` holds a separate "
                "differential-expression analysis of the same GEO series, computed per "
                f"sequencing batch and published as `{cc['reference']['table']}` over a "
                f"{cc['reference']['reference_gene_panel']:,}-gene panel. Its per-batch "
                "results are pooled the same way ours are, as "
                f"`{cc['reference']['pooling']}`, and then compared entry by entry on "
                "log2 fold changes for every (cell line, perturbation, gene) triple both "
                "sides share. The comparison was pre-registered with a gate: pooled "
                f"Spearman at least {C.CROSSCHECK_MIN_SPEARMAN:.2f}, with the number of "
                "compared pairs reported alongside it. The same module also compares our "
                "two control estimators against each other, which is the measurement "
                "that says how much the batch-matched correction actually changes."
            ),
            Head(3, "Results and what they imply"),
            Table(
                ["quantity", "value"],
                [
                    ["pooled Spearman against the reference", _fmt(cc["pooled_spearman"])],
                    ["gene-level entries compared", f"{cc['n_pooled_entries']:,}"],
                    [
                        "shared perturbations",
                        f"{cc['n_shared_cell_line_target_pairs']:,} "
                        f"(HepG2 {cc['per_cell_line']['hepg2']['matched_pairs']:,}, "
                        f"Jurkat {cc['per_cell_line']['jurkat']['matched_pairs']:,})",
                    ],
                    [
                        "median agreement within one perturbation",
                        _fmt(cc["median_per_pair_spearman"]),
                    ],
                    [
                        "batch-matched against batch-pooled, log2FC",
                        _fmt(cc["own_estimator_agreement"]["median_pearson_within_vs_pooled_lfc"]),
                    ],
                    [
                        "batch-matched against batch-pooled, delta",
                        _fmt(cc["own_estimator_agreement"]["median_pearson_within_vs_pooled_delta"]),
                    ],
                ],
            ),
            Para(
                "Two independently written pipelines starting from the same raw matrices "
                "agree on the fold changes, which rules out the class of error that "
                "silently reindexes genes, mixes up controls, or normalizes twice. It "
                "does not validate the choices the two pipelines share, so it is a check "
                "on implementation rather than on method."
            ),
        ]
        st_rows = [
            [s, f"{v['n']:,}", _fmt(v["median_spearman"])]
            for s, v in sorted(cc.get("per_pair_spearman_by_stratum", {}).items())
        ]
        xc.blocks += [
            Table(["stratum", "perturbations", "median Spearman vs reference"], st_rows),
            Para(
                "Agreement with the reference rises with effect size in the same way the "
                "self-agreement does, from "
                f"{_fmt(list(sorted(cc['per_pair_spearman_by_stratum'].items()))[0][1]['median_spearman'])} "
                "in the weakest quartile to "
                f"{_fmt(list(sorted(cc['per_pair_spearman_by_stratum'].items()))[-1][1]['median_spearman'])} "
                "in the strongest. The residual disagreement in the weak strata is the "
                "same noise the ceilings measure, seen from the other side: two analyses "
                "of an effect that barely reproduces cannot agree closely about it "
                "either."
            ),
            Para(
                "The two control estimators agree at median Pearson "
                f"{_fmt(cc['own_estimator_agreement']['median_pearson_within_vs_pooled_lfc'])} "
                "on log2 fold changes, so the batch-matched correction is small on most "
                "perturbations. It is kept as the primary estimate because where the two "
                "disagree, the disagreement is exactly the batch-composition confound "
                "that the weighting removes; a later run that prefers the pooled column "
                "can read it from the same file and should say which one it used."
            ),
            FigRef("reference_crosscheck"),
        ]
        S.append(xc)

    # ---------------- models ----------------
    mc = Section(sources=model_src)
    mc.blocks += [
        Head(2, "The four-model response contract"),
        Head(3, "Definition, procedure, and thresholds"),
        Para(
            f"The shared gene universe holds **{cov['universe_size']:,} genes**: measured "
            f"in both cell lines ({cov['shared_measured_genes']:,} genes qualify on that "  # noqa: E501
            "count alone) and addressable in all four model vocabularies. Identifiers "
            "are resolved to Ensembl gene ids through the assay's own var table first, "
            "then HGNC approved symbols, then HGNC aliases and previous symbols, and the "
            "route taken is recorded for every token in "
            "`gene_universe/vocabulary_<model>.parquet`. Round-tripping the "
            f"{cov['identifier_audit']['n_nadig_genes']:,} gene names in the series fails "
            f"{cov['identifier_audit']['ensg_to_symbol_to_ensg_failures']} time in the "
            "Ensembl to symbol to Ensembl direction and "
            f"{cov['identifier_audit']['symbol_to_ensg_to_symbol_failures']} times in the "
            "reverse direction; unmapped vocabulary tokens are counted, not dropped "
            "silently."
        ),
        Para(
            "The demonstration is deliberately narrow: one real non-targeting control "
            f"cell from {LABEL.get(_demo_field(demos, 'cell_line', 'hepg2'), 'HepG2')} "
            f"(barcode `{_demo_field(demos, 'barcode', 'n/a')}`, "
            f"{_demo_input(demos, 'total_counts', 0):,.0f} total counts and "
            f"{_demo_input(demos, 'genes_detected', 0):,.0f} genes detected) is passed to "
            "each model through that model's own published inference path, and two things "
            "are recorded. The **response vector** is the model's native per-gene output "
            "for the universe genes, whose meaning differs per model and is written down "
            "in `contract.json`. The **Jacobian column** is the derivative of every "
            "gene's response with respect to the observed count of one source gene "
            f"(`{_demo_source(demos, 'gene_id')}`, {_demo_source(demos, 'gene_name')}, "
            f"observed count {_demo_source(demos, 'input_count')}), which is the quantity "
            "a later run needs if it wants to ask how one gene's abundance moves the "
            "model's opinion about the others."
        ),
        Para(
            "Each Jacobian column is computed by the most exact route the model's graph "
            "allows and then cross-checked against a second route or step size, with the "
            "Pearson correlation and the relative L2 difference between the two recorded "
            "in `demo.json`. Where the graph is differentiable end to end, the column is "
            "a forward-mode automatic derivative; where the input is discretized or the "
            "autodiff pass does not fit in memory, it is a central finite difference at a "
            "stated step. The step is part of the measurement, not an implementation "
            "detail, so it is recorded with the value."
        ),
        Para(
            "Symmetry is measured rather than assumed. A run that reads one column of a "
            "response matrix and treats the matrix as symmetric is relying on a property "
            "no model documents, so each model was additionally asked for a second "
            "column at the gene its source gene moves most, and the two cross entries "
            "were compared. **A verdict is recorded only when the Jacobian estimate "
            "itself is stable**, meaning Pearson at least 0.99 and relative L2 difference "
            "at most 0.5 between two step sizes; otherwise step sensitivity and genuine "
            "asymmetry are not separable, and the contract says so instead of guessing."
        ),
        Para(
            "**The per-gene values are not comparable across models, and a consuming run "
            "has to do something about it.** The four outputs are four different "
            "quantities: scGPT returns a predicted expression bin value on its 51-bin "
            "training scale, AIDO.Cell returns its masked-language-model regression "
            "target on a log1p CP10K scale, and scPRINT and TranscriptFormer both return "
            "a share of the cell's counts, because their decoders end in a softmax across "
            "the gene positions supplied in the same forward pass. On this one cell those "
            "ranges are "
            + ", ".join(
                f"{m} {_sig(demos[m]['response_summary']['min'])} to "
                f"{_sig(demos[m]['response_summary']['max'])}"
                for m in MODELS
                if m in demos
            )
            + ". A difference or ratio between two models' raw values has no defined "
            "meaning, and for the two softmax models even a single model's values move "
            "when the supplied gene set changes, for reasons that have nothing to do with "
            "biology. A later run therefore has to declare one comparison space and stay "
            "in it: rank the genes within a cell per model, or standardize within a "
            "model, or better, have each model produce its own contrast between a "
            "perturbed and an unperturbed pass in its own units and compare those "
            "contrasts by correlation or rank overlap. It also has to hold the supplied "
            "gene set fixed across the passes it compares."
        ),
        Para(
            "**Asymmetry constrains how a response matrix may be indexed.** The object a "
            "later run builds is directional: entry (i, j) is the change in gene j's "
            "response caused by a change in gene i, and the two indices are not "
            "interchangeable. Where the measurement below is decisive it comes out "
            "asymmetric, so a run may not fill the other half of the matrix by "
            "transposition, may not average the two triangles, and may not use one "
            "column to stand in for the corresponding row. Getting both directions costs "
            "two passes per pair. For TranscriptFormer the restriction is structural: "
            "with a causal mask only the entries where the source precedes the target are "
            "reachable at all, and the rest are exactly zero by construction rather than "
            "small by measurement."
        ),
        FigRef("model_gene_coverage"),
        Head(3, "Results and what they imply"),
    ]

    mrows = []
    for m, c in cov["coverage"].items():
        mrows.append(
            [
                m,
                f"{c['vocabulary_size']:,}",
                c["identifier_kind"],
                "yes" if c["order_is_a_model_constraint"] else "no",
                f"{c['covers_hepg2_measured']:,} / {c['hepg2_measured_total']:,}",
                f"{c['covers_jurkat_measured']:,} / {c['jurkat_measured_total']:,}",
            ]
        )
    mc.blocks += [
        Table(
            [
                "model",
                "vocabulary",
                "identifier",
                "fixed input order",
                "covers HepG2 measured",
                "covers Jurkat measured",
            ],
            mrows,
        ),
        Para(
            "The vocabularies differ by a factor of three and use different identifier "
            "kinds, so the intersection, not any single model's gene list, is what all "
            "four can be asked about. Two of the four also require a fixed input order, "
            "which means the universe has to carry each model's token positions rather "
            "than only its gene ids."
        ),
    ]

    if contracts:
        crows = []
        for m in MODELS:
            c = contracts.get(m)
            if not c:
                continue
            crows.append(
                [
                    m,
                    _first_sentence(c["output_semantics"]),
                    c["information_flow"],
                    "yes" if c["ordering_dependent"] else "no",
                    (demos.get(m, {}).get("jacobian", {}) or {}).get(
                        "method", c["jacobian_method"]
                    ),
                ]
            )
        mc.blocks += [
            Table(
                [
                    "model",
                    "per-gene value",
                    "information flow",
                    "ordering matters",
                    "Jacobian as run",
                ],
                crows,
            ),
            Para(
                "The recorded Jacobian route is the one that actually ran, which is not "
                "always the one the contract declares as preferred: scGPT's forward-mode "
                "pass over "
                f"{cov['universe_size']:,} genes exhausted the GPU, so its column is a "
                "central finite difference. Reading the declared field instead of the "
                "recorded one would misdescribe the number."
            ),
            FigRef("model_responses_on_one_cell"),
        ]

        probes = {
            m: (demos.get(m, {}).get("jacobian", {}) or {}).get("symmetry_probe")
            for m in MODELS
        }
        if any(probes.values()):
            srows = []
            for m in MODELS:
                pr = probes.get(m)
                if not pr or pr.get("verdict") == "not_measured":
                    continue
                srows.append(
                    [
                        m,
                        pr["second_column_method"],
                        _signed(pr["d_target_d_source"]),
                        _signed(pr["d_source_d_target"]),
                        _sig(pr["ratio_d_source_d_target_over_d_target_d_source"]),
                        pr["verdict"].replace("_", " "),
                    ]
                )
            mc.blocks += [
                Table(
                    [
                        "model",
                        "second column route",
                        "d target / d source",
                        "d source / d target",
                        "ratio",
                        "verdict",
                    ],
                    srows,
                )
            ]
            determined = [
                m
                for m in MODELS
                if (probes.get(m) or {}).get("verdict")
                in ("asymmetric", "symmetric_at_this_pair")
            ]
            undetermined = [
                m
                for m in MODELS
                if (probes.get(m) or {}).get("verdict") == "not_determined"
                and (contracts.get(m) or {}).get("response_symmetric") is None
            ]
            structural = [
                m
                for m in MODELS
                if (probes.get(m) or {}).get("verdict") == "not_determined"
                and (contracts.get(m) or {}).get("response_symmetric") is False
            ]
            if determined:
                mc.blocks.append(
                    Para(
                        "Where the measurement is decisive it comes out asymmetric ("
                        + ", ".join(determined)
                        + "), and for scGPT the two cross entries do not even share a "
                        "sign. The response matrix is directional in fact and not only "
                        "in principle, so the indexing rule stated above is a "
                        "requirement rather than a caution."
                    )
                )
            if undetermined:
                mc.blocks.append(
                    Para(
                        "For "
                        + ", ".join(undetermined)
                        + " the finite-difference Jacobian is itself step-dependent on "
                        "this cell, so the contract records symmetry as not determined. "
                        "The practical reading is narrower than it sounds: the direction "
                        "of that column is reproducible, so ranking genes by sensitivity "
                        "is defensible, while anything that depends on the magnitude "
                        "needs a step-size study first."
                    )
                )
            if structural:
                mc.blocks.append(
                    Para(
                        "For "
                        + ", ".join(structural)
                        + " the cross-entry probe is also step-limited, but symmetry is "
                        "settled by the architecture rather than by this measurement, as "
                        "the ordering probe below shows directly."
                    )
                )

            jrows = []
            for m in MODELS:
                jac = demos.get(m, {}).get("jacobian", {}) or {}
                chk = jac.get("check") or {}
                if not chk:
                    continue
                jrows.append(
                    [
                        m,
                        jac.get("method", ""),
                        chk.get("cross_check_method", ""),
                        _sig(chk.get("pearson_primary_vs_cross_check", float("nan"))),
                        _sig(chk.get("relative_l2_difference", float("nan"))),
                        _sig(jac.get("column_l2", float("nan"))),
                    ]
                )
            mc.blocks += [
                Table(
                    [
                        "model",
                        "Jacobian route",
                        "cross-check route",
                        "Pearson",
                        "relative L2 gap",
                        "column L2",
                    ],
                    jrows,
                ),
                Para(
                    "Only scPRINT's column is an exact derivative, agreeing with an "
                    "independent finite difference to a relative L2 difference of "
                    f"{_sig((demos.get('scprint', {}).get('jacobian', {}).get('check') or {}).get('relative_l2_difference', float('nan')))}"
                    ". The other three are finite differences over integer counts, which "
                    "is the honest description of what a discretized or masked input "
                    "allows: the direction of the column reproduces across step sizes, "
                    "its magnitude does not, and TranscriptFormer's per-gene "
                    "sensitivities on this cell are close to numerical noise (column L2 "
                    f"{_sig((demos.get('transcriptformer', {}).get('jacobian', {}) or {}).get('column_l2', float('nan')))}"
                    ")."
                ),
            ]

        tf = demos.get("transcriptformer", {}).get("ordering_probe")
        if tf and "target_share_change_source_ahead" in tf:
            mc.blocks += [
                Para(
                    "The ordering probe settles TranscriptFormer directly. Reading the "
                    "same target gene twice from the same cell with "
                    f"{tf['context_genes_between_them']} context genes between them, "
                    f"raising the source gene's count from {tf['source_count']:.0f} to "
                    f"{tf['source_count_raised_to']:.0f} changes the target's predicted "
                    f"share by {tf['target_share_change_source_ahead']:+.3e} (on a base "
                    f"of {tf['target_share_source_ahead']:.3e}) when the source sits "
                    f"ahead of the target, and by exactly "
                    f"{tf['target_share_change_source_behind']:+.1f} when it sits behind. "
                    "Half of a TranscriptFormer response matrix is therefore not small, "
                    "it is unreachable, and a run that averages the two triangles is "
                    "averaging measurements with structural zeros."
                )
            ]

        mc.blocks.append(Head(3, "Substitutions"))
        mc.blocks.append(
            Para(
                "Where a published inference path could not be run as shipped, the "
                "substitution is recorded in the model's `contract.json` rather than "
                "hidden:"
            )
        )
        for m in MODELS:
            c = contracts.get(m)
            if not c or not c.get("substitutions"):
                continue
            seen: set[str] = set()
            items = []
            for sub in c["substitutions"]:
                key = str(sub).split("(")[0].strip()
                if key in seen:
                    continue
                seen.add(key)
                items.append(str(sub))
            mc.blocks.append(Para(f"**{m}**"))
            mc.blocks.append(Bullets(items))

        if envs:

            def _torch_of(env: dict) -> str:
                for pkg in env.get("pinned_packages", []):
                    name, _, version = str(pkg).partition("==")
                    if name.strip().lower() == "torch":
                        return version.strip() or "?"
                return "?"

            torch_versions = sorted({_torch_of(e) for e in envs.values()})
            erows = []
            for m in MODELS:
                e, d = envs.get(m), demos.get(m, {})
                if not e:
                    continue
                t = d.get("timings_seconds") or {}
                erows.append(
                    [
                        m,
                        e.get("image", "unrecorded"),
                        e.get("device", "?"),
                        f"{t.get('load', float('nan')):.2f} s",
                        f"{t.get('forward', float('nan')):.3f} s",
                        f"{t.get('jacobian', float('nan')):.2f} s",
                    ]
                )
            mc.blocks += [
                Head(3, "The four stacks as run"),
                Para(
                    "The plan called for one pinned environment per model, on the "
                    "assumption that the four dependency sets are mutually "
                    "unsatisfiable. They are not quite: pinning torch first and "
                    "installing the four model packages with `--no-deps`, since each "
                    "would otherwise drag torch back to the version it was built "
                    f"against, gives one stack (`envs/model_stack.sh`) that runs all "
                    f"four. Every model here ran on torch {', '.join(torch_versions)} in "
                    "the same image, each in its own job, and each records its own "
                    "resolved package set in `environment.json`, so a later run can tell "
                    "whether two results came from the same stack."
                ),
                Table(
                    ["model", "image", "device", "checkpoint load", "forward", "Jacobian"],
                    erows,
                ),
            ]
    S.append(mc)

    # ---------------- file contract ----------------
    fc = Section(sources=f"{rel}/src/scfm_pfm/verify.py;{rel}/src/scfm_pfm/paths.py")
    fc.blocks += [
        Head(2, "The file contract later runs read"),
        Code(
            "effects/<cell_line>/effects.npz              delta and log2FC matrices, both estimators\n"
            "effects/<cell_line>/perturbations.parquet    per-perturbation metadata, knockdown, DE, strata, ceilings\n"
            "effects/<cell_line>/genes.parquet            gene index, pooled control level, expressed flag\n"
            "effects/<cell_line>/noise_ceiling_summary.parquet   per-stratum medians and quartiles\n"
            "effects/<cell_line>/topk_overlap_grid.parquet        per-perturbation overlap at every k, with chance and diagnostics\n"
            "effects/<cell_line>/topk_overlap_grid_summary.parquet  per-stratum medians of that grid\n"
            "effects/<cell_line>/split_half_vectors.npz   the two split halves' log2FC and delta vectors\n"
            "effects/<cell_line>/topk_overlap_grid_qc.json  grid config, chance levels, reproduction of the published ceilings\n"
            "effects/<cell_line>/qc.json                  provenance, filter funnel, config, environment\n"
            "crosscheck/summary.json                      agreement with the published reanalysis\n"
            "gene_universe/gene_universe.parquet          one row per shared gene, with each model's token position\n"
            "gene_universe/vocabulary_<model>.parquet     every vocabulary token, its gene id, and how it resolved\n"
            "gene_universe/coverage.json                  coverage and the identifier audit\n"
            "models/<model>/response_vector.parquet       per-gene response on the demonstration cell\n"
            "models/<model>/jacobian_column.parquet       d response / d source gene count\n"
            "models/<model>/contract.json                 the declared response contract\n"
            "models/<model>/environment.json              pinned packages, image, device\n"
            "models/<model>/demo.json                     what was run, timings, checks, symmetry probe\n"
        ),
        Para(
            "`uv run python -m scfm_pfm.verify --all` re-checks every claim on this page "
            "against those files: matrix shapes and finiteness, knockdown and "
            "differential-expression ranges, the four strata, ceilings bounded in (0, 1] "
            "and ordered across strata, the reference Spearman against its gate, "
            "vocabulary positions inside each model's vocabulary, and one finite response "
            "vector per model with its contract fields filled in, including a symmetry "
            "field that is either answered or explicitly measured and inconclusive. For "
            "the overlap grid it also re-derives the chance levels from the expressed-gene "
            "count, re-checks the excess-over-chance definition and the adaptive k on "
            "every row, confirms that the k = "
            f"{C.EFFECTS.topk} column still equals the published `topk_overlap` column, "
            "and recomputes the overlap from the persisted split-half vectors to confirm "
            "they are the vectors that were scored."
        ),
    ]
    S.append(fc)

    # ---------------- provenance ----------------
    pv = Section(sources=f"{rel}/src/scfm_pfm/data/sources.py;{rel}/src/scfm_pfm/models/sources.py")
    prov_rows = [
        [
            "HepG2 matrix",
            qc["hepg2"]["source"]["url"],
            f"md5 {qc['hepg2']['source']['md5']}",
        ],
        [
            "Jurkat matrix",
            qc["jurkat"]["source"]["url"],
            f"md5 {qc['jurkat']['source']['md5']}",
        ],
    ]
    for m, c in cov["coverage"].items():
        p = c["provenance"]
        ref = p.get("revision") or p.get("md5") or ""
        src_name = p.get("repo") or p.get("source") or p.get("bucket", "")
        prov_rows.append([f"{m} vocabulary", src_name, ref])
    for m in MODELS:
        c = contracts.get(m)
        if c:
            prov_rows.append([f"{m} checkpoint", c["checkpoint"], c["checkpoint_revision"]])
    if cc:
        prov_rows.append(
            ["reference release", cc["reference"]["repo"], cc["reference"]["table"]]
        )
    pv.blocks += [
        Head(2, "Provenance"),
        Table(["artifact", "source", "revision or checksum"], prov_rows),
        Para(
            f"Source series: GEO {data_sources.GEO_SERIES}. Seed {C.SEED} throughout. "
            "The configuration lives in `src/scfm_pfm/config.py` and is copied into every "
            "`qc.json`, so a later run can see exactly which thresholds produced these "
            "matrices."
        ),
    ]
    S.append(pv)

    return S, ctx


def _demo_field(demos: dict, key: str, default=""):
    for m in MODELS:
        if m in demos and key in demos[m]:
            return demos[m][key]
    return default


def _demo_input(demos: dict, key: str, default=0):
    for m in MODELS:
        if m in demos and key in (demos[m].get("input") or {}):
            return demos[m]["input"][key]
    return default


def _demo_source(demos: dict, key: str, default=""):
    for m in MODELS:
        src = demos.get(m, {}).get("source_gene") or {}
        if key in src:
            v = src[key]
            return f"{v:,.0f}" if isinstance(v, float) else v
    return default


def ceil_min_cells() -> int:
    from ..data.ceilings import MIN_CELLS_PER_SIDE

    return MIN_CELLS_PER_SIDE


# --------------------------------------------------------------------------
# renderers
# --------------------------------------------------------------------------


def render_html(sections: list[Section], figs: dict, text: dict) -> str:
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_esc(TITLE)}</title>",
        "</head>",
        "<body>",
    ]
    first_chart = True
    fignum = 0
    for sec in sections:
        attr = f' data-sources="{_esc(sec.sources)}"' if sec.sources else ""
        parts.append(f"<section{attr}>")
        for b in sec.blocks:
            if isinstance(b, Head):
                parts.append(f"<h{b.level}>{_inline(b.text)}</h{b.level}>")
            elif isinstance(b, Para):
                parts.append(f"<p>{_inline(b.text)}</p>")
            elif isinstance(b, Bullets):
                parts.append("<ul>")
                parts += [f"<li>{_inline(i)}</li>" for i in b.items]
                parts.append("</ul>")
            elif isinstance(b, Table):
                parts.append("<table>")
                parts.append(
                    "<tr>" + "".join(f"<th>{_inline(h)}</th>" for h in b.header) + "</tr>"
                )
                for r in b.rows:
                    parts.append(
                        "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                    )
                parts.append("</table>")
            elif isinstance(b, Code):
                parts.append(f"<pre><code>{_esc(b.text)}</code></pre>")
            elif isinstance(b, FigRef):
                fig = figs.get(b.name)
                meta = text.get(b.name, {})
                if fig is None:
                    continue
                fignum += 1
                fragment = fig.figure.to_html(
                    full_html=False,
                    include_plotlyjs=True if first_chart else False,
                    config=PLOTLY_CONFIG,
                )
                first_chart = False
                parts.append(
                    f'<section class="figure" data-sources="{_esc(meta.get("sources", ""))}">'
                )
                parts.append(
                    f'<h3 class="figure-title">Figure {fignum}: '
                    f'{_inline(meta.get("title", b.name))}</h3>'
                )
                parts.append(fragment)
                parts.append(
                    f'<p class="figure-description">{_inline(meta.get("description", ""))}</p>'
                )
                parts.append(
                    f'<p class="figure-findings">{_inline(meta.get("findings", ""))}</p>'
                )
                parts.append("</section>")
        parts.append("</section>")
    parts += ["</body>", "</html>"]
    return "\n".join(parts)


def render_markdown(sections: list[Section], figs: dict, text: dict) -> str:
    out: list[str] = []
    fignum = 0
    for sec in sections:
        for b in sec.blocks:
            if isinstance(b, Head):
                out.append("#" * b.level + " " + b.text + "\n")
            elif isinstance(b, Para):
                out.append(b.text + "\n")
            elif isinstance(b, Bullets):
                out.append("\n".join(f"- {i}" for i in b.items) + "\n")
            elif isinstance(b, Table):
                out.append(
                    "| " + " | ".join(b.header) + " |\n|"
                    + "|".join("---" for _ in b.header) + "|\n"
                    + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in b.rows)
                    + "\n"
                )
            elif isinstance(b, Code):
                out.append("```\n" + b.text + "```\n")
            elif isinstance(b, FigRef):
                if b.name not in figs:
                    continue
                fignum += 1
                meta = text.get(b.name, {})
                out.append(
                    f"**Figure {fignum}: {meta.get('title', b.name)}** "
                    f"([interactive render](figures/{b.name}/{b.name}.html), "
                    f"[plot data](figures/{b.name}/data.json))\n"
                )
                out.append(meta.get("description", "") + "\n")
                out.append(meta.get("findings", "") + "\n")
    header = (
        "> This Markdown file is a reading copy. The primary artifact is the "
        "self-contained HTML report, where every figure is embedded inline and "
        "interactive.\n"
    )
    return header + "\n" + "\n".join(out)


def build(
    root: Path,
    html_path: Path,
    md_path: Path,
    figures_dir: Path,
) -> tuple[str, str]:
    rel = f"experiments/{paths.repo_dir().name}"
    figs = figmod.build_all(root, figures_dir)
    sections, ctx = compose(root, rel)
    text = figure_text(ctx, rel)
    page = render_html(sections, figs, text)
    md = render_markdown(sections, figs, text)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page)
    md_path.write_text(md)
    print(f"wrote {html_path} ({len(page):,} bytes)")
    print(f"wrote {md_path} ({len(md):,} bytes)")
    return page, md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--html", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--figures", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else paths.input_root()
    repo = paths.repo_dir()
    html_path = Path(args.html) if args.html else repo / "results" / "data_preparation_report.html"
    md_path = Path(args.md) if args.md else repo / "results" / "data_preparation_report.md"
    figures_dir = Path(args.figures) if args.figures else repo / "figures"
    build(root, html_path, md_path, figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
