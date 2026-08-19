"""Figures for the data-preparation report.

Every figure reads a file that a later run can read too, so nothing here is a
one-off rendering of numbers that live only in this script. Each builder returns
a :class:`Fig`, so the same figure object can be saved as a bundle (this module's
CLI) and embedded inline in the HTML report (``report.build``).

    uv run python -m scfm_pfm.report.figures --root results/artifacts_mirror
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import EDITORIAL_8, add_reference_line, apply_theme, save_figure_bundle

from .. import config as C
from .. import paths
from ..data.ceilings_grid import DIAGONAL_K, K_GRID

LINES = ("hepg2", "jurkat")
LABEL = {"hepg2": "HepG2", "jurkat": "Jurkat"}
STRATA = list(C.STRATUM_NAMES)
MODELS = ("scgpt", "aido_cell", "scprint", "transcriptformer")

# Entity colours are shared across figures and with the checkpoint cards; the
# file is authoritative, so a colour never means one thing in one figure and
# something else in the next.
_COLOR_FILE = "entity_colors.json"
_FALLBACK_COLORS = {
    "HepG2": "#C4650D",
    "Jurkat": "#4E728A",
    "S1_lowest": "#2E6E4E",
    "S2": "#988453",
    "S3": "#B9605B",
    "S4_highest": "#7495AB",
}


def colors(out: Path) -> dict[str, str]:
    for candidate in (out / _COLOR_FILE, paths.repo_dir() / "figures" / _COLOR_FILE):
        if candidate.exists():
            return json.loads(candidate.read_text())
    return dict(_FALLBACK_COLORS)


def _tint(hex_color: str, amount: float = 0.5) -> str:
    """Blend a colour toward white, so a second series of the same entity reads
    as the same entity while still being separable by lightness."""

    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(v + (255 - v) * amount) for v in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


@dataclass
class Fig:
    """One figure: the Plotly object, its plot-ready data, and its alt text."""

    name: str
    figure: go.Figure
    data: Any
    alt: str
    meta: dict = field(default_factory=dict)

    def save(self, out: Path) -> dict:
        save_figure_bundle(self.figure, self.name, data=self.data, alt=self.alt, root=out)
        return self.meta


def _qc(root: Path, cl: str) -> dict:
    return json.loads((root / "effects" / cl / "qc.json").read_text())


def _pert(root: Path, cl: str) -> pd.DataFrame:
    return pd.read_parquet(root / "effects" / cl / "perturbations.parquet")


def fig_funnel(root: Path, out: Path) -> Fig:
    """Where the cells and perturbations go, step by step."""
    col = colors(out)
    rows = []
    for cl in LINES:
        qc = _qc(root, cl)
        for step in qc["filter_funnel"]:
            if "cells" in step:
                rows.append(
                    {
                        "cell_line": LABEL[cl],
                        "step": step["step"].replace("_", " "),
                        "cells": int(step["cells"]),
                        "criterion": step.get("criterion", ""),
                    }
                )
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for cl in LINES:
        sub = df[df.cell_line == LABEL[cl]]
        fig.add_trace(
            go.Bar(
                x=sub.step,
                y=sub.cells,
                name=LABEL[cl],
                marker_color=col[LABEL[cl]],
                hovertemplate="%{x}<br>%{y:,} cells<extra>" + LABEL[cl] + "</extra>",
            )
        )
    fig.update_xaxes(title_text="filter step, in order")
    fig.update_yaxes(title_text="cells remaining")
    apply_theme(fig, height=430)
    return Fig(
        "filter_funnel",
        fig,
        df.to_dict("records"),
        "Cells remaining after each filter step in both cell lines. Per-cell quality "
        "control removes very little; the working set is set by the perturbation-level "
        "thresholds.",
        {"rows": len(df)},
    )


def fig_knockdown(root: Path, out: Path) -> Fig:
    """On-target knockdown, the check that the perturbation did what it claims."""
    col = colors(out)
    frames = [_pert(root, cl).assign(cell_line=LABEL[cl]) for cl in LINES]
    df = pd.concat(frames, ignore_index=True)
    fig = go.Figure()
    for cl in LINES:
        sub = df[(df.cell_line == LABEL[cl]) & df.knockdown.notna()]
        fig.add_trace(
            go.Histogram(
                x=sub.knockdown,
                name=f"{LABEL[cl]} (n={sub.shape[0]:,})",
                marker_color=col[LABEL[cl]],
                opacity=0.65,
                xbins={"start": -0.5, "end": 1.0, "size": 0.02},
            )
        )
    fig.update_layout(barmode="overlay")
    add_reference_line(
        fig,
        x=C.EFFECTS.knockdown_detect_min,
        label=f"detection floor {C.EFFECTS.knockdown_detect_min:.2f}",
    )
    fig.update_xaxes(title_text="on-target knockdown, 1 − (perturbed ÷ control) CP10K")
    fig.update_yaxes(title_text="perturbations")
    apply_theme(fig, height=420)
    medians = {
        LABEL[cl]: float(df[df.cell_line == LABEL[cl]].knockdown.median()) for cl in LINES
    }
    return Fig(
        "knockdown_distribution",
        fig,
        {
            "medians": medians,
            "values": {
                LABEL[cl]: df[df.cell_line == LABEL[cl]].knockdown.dropna().round(4).tolist()
                for cl in LINES
            },
        },
        "Distribution of on-target knockdown per perturbation in both cell lines. "
        f"Medians {medians['HepG2']:.4f} (HepG2) and {medians['Jurkat']:.4f} (Jurkat); "
        "the marked line is the detection floor below which a perturbation is flagged.",
        medians,
    )


def fig_knockdown_vs_effect(root: Path, out: Path) -> Fig:
    """Knockdown of the target against the transcriptome-wide response."""
    col = colors(out)
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=[LABEL[cl] for cl in LINES],
        shared_yaxes=True, horizontal_spacing=0.07,
    )
    summary = {}
    for i, cl in enumerate(LINES, start=1):
        p = _pert(root, cl)
        sub = p[p.knockdown.notna()]
        fig.add_trace(
            go.Scattergl(
                x=sub.knockdown,
                y=sub.effect_l2_lfc,
                mode="markers",
                name=LABEL[cl],
                showlegend=False,
                marker={"size": 4, "opacity": 0.45, "color": col[LABEL[cl]]},
                text=sub.target,
                hovertemplate="%{text}<br>knockdown %{x:.3f}<br>effect L2 %{y:.2f}"
                "<extra></extra>",
            ),
            row=1, col=i,
        )
        add_reference_line(
            fig, x=C.EFFECTS.knockdown_detect_min,
            label=f"detection floor {C.EFFECTS.knockdown_detect_min:.2f}",
            row=1, col=i,
        )
        fig.update_xaxes(title_text="on-target knockdown", row=1, col=i)
        summary[LABEL[cl]] = {
            "n": int(sub.shape[0]),
            "spearman_knockdown_vs_effect": float(
                sub.knockdown.corr(sub.effect_l2_lfc, method="spearman")
            ),
            "median_effect_l2_above_floor": float(
                sub[sub.knockdown >= C.EFFECTS.knockdown_detect_min].effect_l2_lfc.median()
            ),
            "median_effect_l2_below_floor": float(
                sub[sub.knockdown < C.EFFECTS.knockdown_detect_min].effect_l2_lfc.median()
            ),
        }
    fig.update_yaxes(title_text="effect size, L2 norm of log2FC", row=1, col=1)
    apply_theme(fig, height=430)
    return Fig(
        "knockdown_vs_effect",
        fig,
        summary,
        "Each retained perturbation's on-target knockdown against the size of its "
        "transcriptome-wide response. Strong knockdown does not imply a large "
        "downstream response: at knockdown above the detection floor the effect sizes "
        "span the full observed range.",
        summary,
    )


def fig_effect_size(root: Path, out: Path) -> Fig:
    """Effect size against differential expression, coloured by stratum."""
    col = colors(out)
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=[LABEL[cl] for cl in LINES],
        shared_yaxes=True, horizontal_spacing=0.08,
    )
    summary = {}
    for i, cl in enumerate(LINES, start=1):
        p = _pert(root, cl)
        for stratum in STRATA:
            sub = p[p.effect_stratum == stratum]
            fig.add_trace(
                go.Scattergl(
                    x=sub.effect_l2_lfc,
                    y=sub.n_de_genes + 1,
                    mode="markers",
                    name=stratum,
                    legendgroup=stratum,
                    showlegend=i == 1,
                    marker={"size": 4, "opacity": 0.5, "color": col[stratum]},
                    text=sub.target,
                    hovertemplate="%{text}<br>effect L2 %{x:.2f}<br>%{y} DE genes + 1"
                    "<extra></extra>",
                ),
                row=1, col=i,
            )
        summary[LABEL[cl]] = {
            s: {
                "n": int((p.effect_stratum == s).sum()),
                "median_de": float(p[p.effect_stratum == s].n_de_genes.median()),
                "median_effect_l2": float(p[p.effect_stratum == s].effect_l2_lfc.median()),
                "min_effect_l2": float(p[p.effect_stratum == s].effect_l2_lfc.min()),
                "max_effect_l2": float(p[p.effect_stratum == s].effect_l2_lfc.max()),
            }
            for s in STRATA
        }
        fig.update_xaxes(title_text="effect size, L2 norm of log2FC", row=1, col=i)
    fig.update_yaxes(
        title_text="differentially expressed genes + 1", type="log", row=1, col=1
    )
    apply_theme(fig, height=440)
    return Fig(
        "effect_size_vs_de",
        fig,
        summary,
        "Effect size against the number of differentially expressed genes for every "
        "retained perturbation, coloured by effect-size stratum. The strata are "
        "equal-count quartiles of effect size within each cell line, so the colours "
        "also show where the quartile boundaries fall. The y-axis is logarithmic and "
        "shifted by one so perturbations with no differentially expressed gene appear.",
        summary,
    )


def fig_ceilings(root: Path, out: Path) -> Fig:
    """The noise ceiling: how well a perturbation reproduces itself."""
    col = colors(out)
    frames = [
        pd.read_parquet(root / "effects" / cl / "noise_ceiling_summary.parquet").assign(
            cell_line=LABEL[cl]
        )
        for cl in LINES
    ]
    df = pd.concat(frames, ignore_index=True)
    metrics = list(C.CEILING_METRICS)
    fig = make_subplots(
        rows=1, cols=len(metrics), subplot_titles=metrics, shared_yaxes=True,
        horizontal_spacing=0.06,
    )
    for j, metric in enumerate(metrics, start=1):
        for cl in LINES:
            for split in C.CEILING_SPLITS:
                sub = df[
                    (df.cell_line == LABEL[cl])
                    & (df.metric == metric)
                    & (df.split == split)
                ]
                sub = (
                    sub[sub.stratum.isin(STRATA)]
                    .set_index("stratum")
                    .reindex(STRATA)
                    .reset_index()
                )
                fig.add_trace(
                    go.Bar(
                        x=sub.stratum,
                        y=sub["median"],
                        name=f"{LABEL[cl]}, {split.replace('_', ' ')}",
                        legendgroup=f"{cl}{split}",
                        showlegend=j == 1,
                        marker_color=(
                            col[LABEL[cl]]
                            if split == "split_half_cells"
                            else _tint(col[LABEL[cl]], 0.55)
                        ),
                        marker_pattern_shape="" if split == "split_half_cells" else "/",
                        hovertemplate="%{x}<br>median %{y:.4f}<extra></extra>",
                    ),
                    row=1, col=j,
                )
        fig.update_xaxes(title_text="effect-size stratum", row=1, col=j)
    fig.update_yaxes(title_text="median replicate agreement", row=1, col=1)
    apply_theme(fig, height=440)
    return Fig(
        "noise_ceilings",
        fig,
        df[df.stratum.isin(STRATA + ["all"])].to_dict("records"),
        "Median agreement of each perturbation with itself when its cells are split in "
        "two, by effect-size stratum, metric, cell line and split form. Hatched bars "
        "are the split by sequencing batch, solid bars the split by cell. This is the "
        "level a model prediction of these effects is measured against.",
        {"rows": int(df.shape[0])},
    )


def fig_topk_grid(root: Path, out: Path) -> Fig:
    """Top-k overlap over the whole k grid, against the chance level it has to beat."""
    col = colors(out)
    split = "split_half_cells"
    k_labels = [str(k) for k in K_GRID]
    frames = {
        cl: pd.read_parquet(root / "effects" / cl / "topk_overlap_grid_summary.parquet")
        for cl in LINES
    }
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True,
        subplot_titles=[LABEL[cl] for cl in LINES] + ["", ""],
        vertical_spacing=0.08, horizontal_spacing=0.07,
    )
    data = {}
    for i, cl in enumerate(LINES, start=1):
        df = frames[cl]
        one = df[df.split == split]
        fixed = one[one.k_label.isin(k_labels)].assign(k=lambda x: x.k_label.astype(int))
        adaptive = one[one.k_label == "adaptive"].set_index("stratum")
        for stratum in STRATA:
            s = fixed[fixed.stratum == stratum].sort_values("k")
            for row, ycol, name in (
                (1, "median_overlap", "overlap"),
                (2, "median_excess_over_chance", "excess"),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=s.k, y=s[ycol], mode="lines+markers", name=stratum,
                        legendgroup=stratum, showlegend=(i == 1 and row == 1),
                        line={"color": col[stratum], "width": 2},
                        marker={"color": col[stratum], "size": 7},
                        hovertemplate=f"{stratum}<br>k %{{x}}<br>median {name} "
                        "%{y:.4f}<extra></extra>",
                    ),
                    row=row, col=i,
                )
        chance = fixed[fixed.stratum == "all"].sort_values("k")
        fig.add_trace(
            go.Scatter(
                x=chance.k, y=chance.chance, mode="lines", name="chance, k/n",
                legendgroup="chance", showlegend=i == 1,
                line={"color": EDITORIAL_8[7], "width": 2, "dash": "dash"},
                hovertemplate="k %{x}<br>chance %{y:.4f}<extra></extra>",
            ),
            row=1, col=i,
        )
        diag = pd.DataFrame(
            [
                fixed[(fixed.stratum == s) & (fixed.k == k)].iloc[0]
                for s, k in DIAGONAL_K.items()
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=diag.k, y=diag.median_overlap, mode="markers",
                name="stratum's own k", legendgroup="diagonal", showlegend=i == 1,
                marker={
                    "symbol": "square-open", "size": 15,
                    "color": [col[s] for s in diag.stratum], "line": {"width": 2},
                },
                text=diag.stratum,
                hovertemplate="%{text}<br>k %{x}<br>median overlap %{y:.4f}<extra></extra>",
            ),
            row=1, col=i,
        )
        ad = adaptive.reindex(STRATA).reset_index()
        fig.add_trace(
            go.Scatter(
                x=ad.median_k, y=ad.median_overlap, mode="markers",
                name="adaptive k", legendgroup="adaptive", showlegend=i == 1,
                marker={
                    "symbol": "x", "size": 9, "color": [col[s] for s in ad.stratum],
                },
                text=ad.stratum,
                hovertemplate="%{text}<br>median adaptive k %{x:.0f}<br>"
                "median overlap %{y:.4f}<extra></extra>",
            ),
            row=1, col=i,
        )
        fig.update_xaxes(
            type="log", tickvals=list(K_GRID), ticktext=[str(k) for k in K_GRID],
            title_text="genes kept per side, k", row=2, col=i,
        )
        fig.update_xaxes(type="log", tickvals=list(K_GRID), row=1, col=i)
        data[LABEL[cl]] = one.to_dict("records")
    add_reference_line(fig, y=0.0, label="chance", row=2, col="all")
    fig.update_yaxes(title_text="median top-k overlap", row=1, col=1)
    fig.update_yaxes(title_text="median excess over chance", row=2, col=1)
    apply_theme(fig, height=660)
    return Fig(
        "topk_overlap_grid",
        fig,
        data,
        "Median overlap of the two split halves' top-k gene sets, over k = 10, 50, 100 "
        "and 500 (log axis), by effect-size stratum and cell line. The dashed line is "
        "the chance level k/n for n expressed genes, the open square is the k the "
        "report quotes for that stratum, and the cross is the median adaptive k, which "
        "is each perturbation's own differential-expression count clipped to [10, 500]. "
        "The lower panels rescale the same numbers as excess over chance, where 0 is "
        "chance and 1 is a perfectly reproduced set. Raw overlap climbs with k largely "
        "because chance climbs with k; the excess does not.",
        {"split": split, "k_grid": list(K_GRID)},
    )


def fig_ceiling_vs_cells(root: Path, out: Path) -> Fig:
    """The same ceiling against how many cells the perturbation actually has."""
    col = colors(out)
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=[LABEL[cl] for cl in LINES],
        shared_yaxes=True, horizontal_spacing=0.07,
    )
    summary = {}
    for i, cl in enumerate(LINES, start=1):
        p = _pert(root, cl)
        sub = p[p.split_half_cells__pearson_lfc.notna()]
        for stratum in STRATA:
            s = sub[sub.effect_stratum == stratum]
            fig.add_trace(
                go.Scattergl(
                    x=s.n_cells,
                    y=s.split_half_cells__pearson_lfc,
                    mode="markers",
                    name=stratum,
                    legendgroup=stratum,
                    showlegend=i == 1,
                    marker={"size": 4, "opacity": 0.45, "color": col[stratum]},
                    text=s.target,
                    hovertemplate="%{text}<br>%{x} cells<br>split-half r %{y:.3f}"
                    "<extra></extra>",
                ),
                row=1, col=i,
            )
        bins = pd.qcut(sub.n_cells, 6, duplicates="drop")
        trend = sub.groupby(bins, observed=True).agg(
            cells=("n_cells", "median"),
            r=("split_half_cells__pearson_lfc", "median"),
            n=("n_cells", "size"),
        )
        fig.add_trace(
            go.Scatter(
                x=trend.cells,
                y=trend.r,
                mode="lines+markers",
                name="median by cell-count bin",
                legendgroup="trend",
                showlegend=i == 1,
                line={"color": EDITORIAL_8[7], "width": 3},
                marker={"color": EDITORIAL_8[7], "size": 8},
                hovertemplate="%{x:.0f} cells<br>median r %{y:.3f}<extra></extra>",
            ),
            row=1, col=i,
        )
        fig.update_xaxes(title_text="cells in the perturbation (log)", type="log", row=1, col=i)
        summary[LABEL[cl]] = {
            "n": int(sub.shape[0]),
            "spearman_cells_vs_ceiling": float(
                sub.n_cells.corr(sub.split_half_cells__pearson_lfc, method="spearman")
            ),
            "trend": [
                {"median_cells": float(r.cells), "median_r": float(r.r), "n": int(r.n)}
                for r in trend.itertuples()
            ],
        }
    fig.update_yaxes(title_text="split-half agreement on log2FC", row=1, col=1)
    apply_theme(fig, height=430)
    return Fig(
        "ceiling_vs_cells",
        fig,
        summary,
        "Per-perturbation split-half agreement against the number of cells the "
        "perturbation has, coloured by effect-size stratum, with the median per "
        "cell-count bin drawn over it. Cell count alone does not buy reproducibility: "
        "the effect size matters more than the depth at this range.",
        summary,
    )


def fig_crosscheck(root: Path, out: Path) -> Fig:
    """Our estimates against the published reference, and against each other."""
    col = colors(out)
    cc = root / "crosscheck"
    summary = json.loads((cc / "summary.json").read_text())
    pairs = pd.read_parquet(cc / "reference_agreement.parquet")
    own = pd.read_parquet(cc / "own_estimator_agreement.parquet")
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.1,
        subplot_titles=[
            "per-gene log2FC against the reference",
            "agreement per perturbation",
        ],
    )
    scatter_path = cc / "scatter_sample.parquet"
    if scatter_path.exists():
        pts = pd.read_parquet(scatter_path)
        fig.add_trace(
            go.Scattergl(
                x=pts.our_log2fc, y=pts.reference_log2fc, mode="markers",
                marker={"size": 3, "opacity": 0.35, "color": col["HepG2"]},
                name="genes", text=pts.target,
                hovertemplate="%{text}<br>ours %{x:.3f}<br>reference %{y:.3f}<extra></extra>",
            ),
            row=1, col=1,
        )
        lo = float(min(pts.our_log2fc.min(), pts.reference_log2fc.min()))
        hi = float(max(pts.our_log2fc.max(), pts.reference_log2fc.max()))
        fig.add_trace(
            go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                line={"dash": "dash", "color": col["S1_lowest"]},
            ),
            row=1, col=1,
        )
    fig.add_trace(
        go.Histogram(
            x=pairs.spearman, name="vs published reference", opacity=0.7, nbinsx=50,
            marker_color=col["HepG2"],
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=own.pearson_lfc_within_vs_pooled,
            name="batch-matched vs batch-pooled", opacity=0.7, nbinsx=50,
            marker_color=col["Jurkat"],
        ),
        row=1, col=2,
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="our log2 fold change", row=1, col=1)
    fig.update_yaxes(title_text="reference log2 fold change", row=1, col=1)
    fig.update_xaxes(title_text="correlation across genes", row=1, col=2)
    fig.update_yaxes(title_text="perturbations", row=1, col=2)
    apply_theme(fig, height=440)
    return Fig(
        "reference_crosscheck",
        fig,
        {
            "summary": summary,
            "per_pair_spearman_quantiles": pairs.spearman.quantile(
                [0.05, 0.25, 0.5, 0.75, 0.95]
            ).round(4).to_dict(),
        },
        "Left: our per-gene log2 fold changes against the published reference release, "
        f"pooled Spearman {summary['pooled_spearman']:.4f} over "
        f"{summary['n_pooled_entries']:,} entries, with the identity line. Right: "
        "per-perturbation agreement, ours against the reference and our two control "
        "estimators against each other.",
        summary,
    )


def fig_model_coverage(root: Path, out: Path) -> Fig:
    """What each model's vocabulary can say about this assay."""
    cov = json.loads((root / "gene_universe" / "coverage.json").read_text())
    rows = []
    for model, c in cov["coverage"].items():
        rows.append(
            {
                "model": model,
                "vocabulary": int(c["vocabulary_size"]),
                "resolved to Ensembl": int(c["n_resolved_to_distinct_ensembl"]),
                "covers HepG2 measured": int(c["covers_hepg2_measured"]),
                "covers Jurkat measured": int(c["covers_jurkat_measured"]),
                "identifier_kind": c["identifier_kind"],
                "order_is_a_model_constraint": c["order_is_a_model_constraint"],
            }
        )
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for column in (
        "vocabulary",
        "resolved to Ensembl",
        "covers HepG2 measured",
        "covers Jurkat measured",
    ):
        fig.add_trace(
            go.Bar(
                x=df.model, y=df[column], name=column,
                hovertemplate="%{x}<br>" + column + ": %{y:,}<extra></extra>",
            )
        )
    add_reference_line(
        fig, y=cov["universe_size"],
        label=f"shared universe {cov['universe_size']:,} genes",
    )
    fig.update_xaxes(title_text="model")
    fig.update_yaxes(title_text="genes (log)", type="log")
    apply_theme(fig, height=430)
    return Fig(
        "model_gene_coverage",
        fig,
        {
            "per_model": rows,
            "universe_size": cov["universe_size"],
            "shared_measured_genes": cov["shared_measured_genes"],
            "identifier_audit": cov["identifier_audit"],
        },
        "Vocabulary size, tokens resolved to a distinct Ensembl gene id, and coverage "
        "of the genes measured in each cell line, for the four models. The marked line "
        "is the shared universe every model can address. The axis is logarithmic "
        "because the vocabularies differ by a factor of three.",
        {"universe_size": cov["universe_size"], "models": len(rows)},
    )


def fig_model_responses(root: Path, out: Path) -> Fig:
    """The four response vectors on the same cell, on their own scales."""
    models = [
        m for m in MODELS if (root / "models" / m / "response_vector.parquet").exists()
    ]
    fig = make_subplots(rows=1, cols=max(len(models), 1), subplot_titles=models,
                        horizontal_spacing=0.05)
    summary = {}
    for i, m in enumerate(models, start=1):
        r = pd.read_parquet(root / "models" / m / "response_vector.parquet")
        meta = json.loads((root / "models" / m / "demo.json").read_text())
        expressed = r.input_count > 0
        fig.add_trace(
            go.Scattergl(
                x=np.log1p(r.input_count), y=r.response, mode="markers",
                marker={"size": 3, "opacity": 0.3},
                name=m, showlegend=False, text=r.gene_name,
                hovertemplate="%{text}<br>log1p(count) %{x:.2f}<br>response %{y:.4g}"
                "<extra></extra>",
            ),
            row=1, col=i,
        )
        fig.update_xaxes(title_text="log1p(observed count)", row=1, col=i)
        summary[m] = {
            "spearman_response_vs_input": meta["response_summary"][
                "spearman_response_vs_input"
            ],
            "median_expressed": float(r.response[expressed].median()),
            "median_unexpressed": float(r.response[~expressed].median()),
            "output_semantics": json.loads(
                (root / "models" / m / "contract.json").read_text()
            )["output_semantics"],
        }
    fig.update_yaxes(title_text="model response (own scale)", row=1, col=1)
    apply_theme(fig, height=420)
    return Fig(
        "model_responses_on_one_cell",
        fig,
        summary,
        "Each model's per-gene response on the same real control cell, against the "
        "observed count of that gene. The vertical scales are not comparable across "
        "panels: the four models emit different quantities, which is the reason the "
        "contract records what each one means.",
        summary,
    )


BUILDERS: dict[str, Callable[[Path, Path], Fig]] = {
    "filter_funnel": fig_funnel,
    "knockdown_distribution": fig_knockdown,
    "knockdown_vs_effect": fig_knockdown_vs_effect,
    "effect_size_vs_de": fig_effect_size,
    "reference_crosscheck": fig_crosscheck,
    "noise_ceilings": fig_ceilings,
    "topk_overlap_grid": fig_topk_grid,
    "ceiling_vs_cells": fig_ceiling_vs_cells,
    "model_gene_coverage": fig_model_coverage,
    "model_responses_on_one_cell": fig_model_responses,
}


def build_all(root: Path, out: Path, only: list[str] | None = None) -> dict[str, Fig]:
    """Build every figure and return the objects, without saving them."""
    figs: dict[str, Fig] = {}
    for name, fn in BUILDERS.items():
        if only and name not in only:
            continue
        try:
            figs[name] = fn(root, out)
        except FileNotFoundError as exc:
            print(f"[figure] {name}: skipped ({exc})", flush=True)
    return figs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None, help="directory the figure bundles land in")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else paths.input_root()
    out = Path(args.out) if args.out else paths.repo_dir() / "figures"
    results = {}
    for name, fig in build_all(root, out, args.only).items():
        results[name] = fig.save(out)
        print(f"[figure] {name}: ok", flush=True)
    print(json.dumps(results, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
