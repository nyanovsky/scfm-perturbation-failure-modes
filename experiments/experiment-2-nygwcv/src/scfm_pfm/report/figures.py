"""Every figure in run 2's report, built from the artifacts themselves.

Each function returns a :class:`Fig`, which saves three things side by side: the
interactive HTML bundle with its plot-ready ``data.json`` (so a figure can be
replotted without rerunning extraction), and a vector PDF plus a high-resolution
PNG for publication use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle

from .. import config as C
from .. import paths

LINES = ("hepg2", "jurkat")
LABEL = {"hepg2": "HepG2", "jurkat": "Jurkat"}
MODELS = ("scgpt", "aido_cell", "scprint", "transcriptformer")
MODEL_LABEL = {
    "scgpt": "scGPT",
    "aido_cell": "AIDO.Cell",
    "scprint": "scPRINT",
    "transcriptformer": "TranscriptFormer",
}
ANCHOR_LABEL = {
    "uniform_virtual": "uniform virtual",
    "mean_control": "mean control",
    "individual_cells": "individual cells",
    "mean_control_masked": "mean control, masked",
}
PNG_SCALE = 3

#: run 1 mapped the cell lines and the effect strata; run 2 continues from the
#: next unused editorial slots and adds the four models
_ENTITY_COLORS = {
    "HepG2": "#C4650D",
    "Jurkat": "#4E728A",
    "S1_lowest": "#2E6E4E",
    "S2": "#988453",
    "S3": "#B9605B",
    "S4_highest": "#7495AB",
    "scPRINT": "#2E6E4E",
    "scGPT": "#988453",
    "TranscriptFormer": "#B9605B",
    "AIDO.Cell": "#7495AB",
    "ceiling": "#C4650D",
    "mean-effect baseline": "#4E728A",
    "cross-perturbation": "#2E6E4E",
}


def colors(out: Path) -> dict[str, str]:
    """The shared entity-colour mapping, extending whatever already exists.

    The checkpoint agent assigns colours as it renders cards, and run 1 mapped
    the cell lines and effect strata. Both files win over anything here, so a
    model keeps one colour across the timeline and the report.
    """
    merged = dict(_ENTITY_COLORS)
    for prior in (
        paths.repo_dir() / "figures" / "checkpoint-renders" / "entity_colors.json",
        out / "figures" / "entity_colors.json",
        out / "entity_colors.json",
    ):
        if prior.exists():
            merged.update(json.loads(prior.read_text()))
    path = out / "figures" / "entity_colors.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2))
    return merged


@dataclass
class Fig:
    name: str
    figure: go.Figure
    data: Any
    alt: str
    meta: dict = field(default_factory=dict)

    def save(self, out: Path) -> dict:
        target = Path(
            save_figure_bundle(
                self.figure, self.name, data=self.data, alt=self.alt, root=out / "figures"
            )
        )
        try:
            self.figure.write_image(str(target / f"{self.name}.pdf"), format="pdf")
            self.figure.write_image(
                str(target / f"{self.name}.png"), format="png", scale=PNG_SCALE
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[figures] static export failed for {self.name}: {exc}", flush=True)
        return self.meta


def lighten(hex_color: str, amount: float = 0.62) -> str:
    """Mix a hex colour toward white, so a null series reads as a pale sibling."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(c + (255 - c) * amount) for c in rgb]
    return "#" + "".join(f"{c:02X}" for c in mixed)


def _panel_grid(rows: int, cols: int, titles: list[str], **kw) -> go.Figure:
    return make_subplots(rows=rows, cols=cols, subplot_titles=titles, **kw)


# --------------------------------------------------------------------------- #
# the expression floor
# --------------------------------------------------------------------------- #
def fig_floor_ceiling(summary: pd.DataFrame, out: Path) -> Fig:
    """Replicate ceiling and read-out coverage, against the expression floor."""
    col = colors(out)
    sub = summary[summary.split == "split_half_cells"]
    fig = _panel_grid(
        2,
        2,
        [f"{LABEL[cl]}: ceiling" for cl in LINES] + [f"{LABEL[cl]}: genes kept" for cl in LINES],
        shared_xaxes=True,
        vertical_spacing=0.12,
    )
    rows = []
    for ci, cl in enumerate(LINES, start=1):
        for stratum in ("S3", "S4_highest"):
            s = sub[(sub.cell_line == cl) & (sub.effect_stratum == stratum)].sort_values(
                "floor_cp10k"
            )
            fig.add_trace(
                go.Scatter(
                    x=s.floor_cp10k,
                    y=s.median_pearson_lfc,
                    name=stratum.replace("_", " ") if ci == 1 else None,
                    legendgroup=stratum,
                    showlegend=ci == 1,
                    mode="lines+markers",
                    line=dict(color=col[stratum]),
                    marker=dict(color=col[stratum]),
                    hovertemplate="floor %{x} CP10K<br>ceiling %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=ci,
            )
            rows += [
                {
                    "cell_line": cl,
                    "stratum": stratum,
                    "floor_cp10k": float(r.floor_cp10k),
                    "ceiling": float(r.median_pearson_lfc),
                    "n_genes": int(r.n_expressed_genes),
                }
                for r in s.itertuples()
            ]
        g = sub[(sub.cell_line == cl) & (sub.effect_stratum == "S4_highest")].sort_values(
            "floor_cp10k"
        )
        fig.add_trace(
            go.Scatter(
                x=g.floor_cp10k,
                y=g.n_expressed_genes,
                name=LABEL[cl],
                legendgroup=cl,
                showlegend=False,
                mode="lines+markers",
                line=dict(color=col[LABEL[cl]]),
                marker=dict(color=col[LABEL[cl]]),
                hovertemplate="floor %{x} CP10K<br>%{y:,} genes<extra></extra>",
            ),
            row=2,
            col=ci,
        )
    for c in (1, 2):
        fig.update_xaxes(title_text="expression floor (CP10K, log)", type="log", row=2, col=c)
        fig.update_yaxes(title_text="genes above floor (log)", type="log", row=2, col=c)
        fig.update_yaxes(title_text="median Pearson ceiling", range=[0, 1], row=1, col=c)
    apply_theme(fig, height=620)
    return Fig(
        name="floor_ceiling",
        figure=fig,
        data=rows,
        alt="Ceiling rises and gene count falls as the expression floor rises, in both cell lines.",
        meta={
            "title": "Replicate ceiling and read-out coverage against the expression floor",
            "description": "Top row: median per-perturbation Pearson correlation between the two "
            "replicate halves' log2 fold-change vectors, over genes above each floor, for the two "
            "in-scope effect strata. Bottom row: how many genes survive that floor. Split form "
            "split_half_cells; the batch split agrees to within 0.002.",
            "findings": "The ceiling never stops rising, so no floor satisfies a plateau rule, and "
            "each step up costs most of the read-out set.",
            "sources": "experiments/experiment-2-nygwcv/results/floor_sweep/floor_ceiling_summary.parquet",
        },
    )


def fig_floor_shared_response(shared: pd.DataFrame, out: Path) -> Fig:
    """Is a higher floor buying its ceiling from the shared response?"""
    col = colors(out)
    sub = shared[shared.split == "split_half_cells"]
    strata = list(C.PERTURBED_STRATA)
    fig = _panel_grid(
        len(strata),
        2,
        [f"{LABEL[cl]}, {s.replace('_', ' ')}" for s in strata for cl in LINES],
        shared_xaxes=True,
        vertical_spacing=0.12,
    )
    series = (
        ("median_ceiling", "ceiling"),
        ("median_loo_mean_baseline", "mean-effect baseline"),
        ("median_cross_perturbation_pearson", "cross-perturbation"),
    )
    rows = []
    for ri, stratum in enumerate(strata, start=1):
        for ci, cl in enumerate(LINES, start=1):
            s = sub[(sub.cell_line == cl) & (sub.effect_stratum == stratum)].sort_values(
                "floor_cp10k"
            )
            for key, label in series:
                fig.add_trace(
                    go.Scatter(
                        x=s.floor_cp10k,
                        y=s[key],
                        name=label if (ri == 1 and ci == 1) else None,
                        legendgroup=label,
                        showlegend=ri == 1 and ci == 1,
                        mode="lines+markers",
                        line=dict(color=col[label]),
                        marker=dict(color=col[label]),
                        hovertemplate="floor %{x} CP10K<br>%{y:.4f}<extra></extra>",
                    ),
                    row=ri,
                    col=ci,
                )
            rows += [
                {
                    "cell_line": cl,
                    "stratum": stratum,
                    "floor_cp10k": float(r.floor_cp10k),
                    "ceiling": float(r.median_ceiling),
                    "loo_mean_baseline": float(r.median_loo_mean_baseline),
                    "cross_perturbation": float(r.median_cross_perturbation_pearson),
                    "pc1_share": float(r.pc1_variance_share_uncentred),
                }
                for r in s.itertuples()
            ]
    for ci in (1, 2):
        fig.update_xaxes(
            title_text="expression floor (CP10K, log)", type="log", row=len(strata), col=ci
        )
        for ri in range(1, len(strata) + 1):
            fig.update_yaxes(title_text="Pearson", range=[0, 1], row=ri, col=ci)
    apply_theme(fig, height=620)
    # read the headline off the same rows the figure draws, so the caption cannot drift
    # from the curve
    at_floor = [
        r for r in rows
        if r["cell_line"] == "hepg2" and r["stratum"] == C.FLOOR_RULE_STRATUM
        and abs(r["floor_cp10k"] - C.RUN1_FLOOR_CP10K) < 1e-9
    ]
    share = 100 * at_floor[0]["loo_mean_baseline"] / at_floor[0]["ceiling"] if at_floor else float("nan")
    band = at_floor[0]["ceiling"] - at_floor[0]["loo_mean_baseline"] if at_floor else float("nan")
    return Fig(
        name="floor_shared_response",
        figure=fig,
        data=rows,
        alt="The mean-effect baseline stays nearly flat while the ceiling rises with the floor.",
        meta={
            "title": "How much of the ceiling a stratum-average predictor already reaches",
            "description": "Ceiling as in the previous figure; mean-effect baseline predicts each "
            "perturbation by the average of the other perturbations in its stratum, built from one "
            "replicate half and scored against the other; cross-perturbation is the median Pearson "
            "between different perturbations across the two halves.",
            "findings": "The baseline is nearly flat across the floor grid, so the extra "
            "correlation a higher floor buys is mostly perturbation-specific. At the inclusive "
            f"floor the baseline already reaches {share:.3g}% of the HepG2 S4 ceiling, leaving "
            f"{band:.4g} of correlation for everything perturbation-specific.",
            "sources": "experiments/experiment-2-nygwcv/results/floor_sweep/floor_shared_response.parquet",
        },
    )


# --------------------------------------------------------------------------- #
# extraction diagnostics
# --------------------------------------------------------------------------- #
def fig_step_stability(bundles: dict, out: Path) -> Fig:
    """Adjacent-step agreement of the finite-difference column, per model."""
    col = colors(out)
    anchors = [a for a in C.ANCHORS]
    fig = _panel_grid(
        len(anchors),
        2,
        [f"{LABEL[cl]}, {ANCHOR_LABEL[a]}" for a in anchors for cl in LINES],
        shared_xaxes=True,
        vertical_spacing=0.09,
    )
    rows = []
    for ri, anchor in enumerate(anchors, start=1):
        for ci, cl in enumerate(LINES, start=1):
            for model in MODELS:
                b = bundles.get((model, cl))
                if not b:
                    continue
                frame = pd.DataFrame(b["step_sweep"])
                if frame.empty or "anchor" not in frame:
                    continue
                s = frame[frame.anchor == anchor].sort_values("step")
                if s.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=s.step,
                        y=s.median_pearson_adjacent,
                        name=MODEL_LABEL[model] if (ri == 1 and ci == 1) else None,
                        legendgroup=model,
                        showlegend=ri == 1 and ci == 1,
                        mode="lines+markers",
                        line=dict(color=col[MODEL_LABEL[model]]),
                        marker=dict(color=col[MODEL_LABEL[model]]),
                        hovertemplate="step %{x}<br>adjacent Pearson %{y:.5f}<extra></extra>",
                    ),
                    row=ri,
                    col=ci,
                )
                rows += [
                    {
                        "model": model,
                        "cell_line": cl,
                        "anchor": anchor,
                        "step": float(r.step),
                        "next_step": float(r.next_step),
                        "median_pearson_adjacent": float(r.median_pearson_adjacent),
                        "median_relative_l2_adjacent": float(r.median_relative_l2_adjacent),
                    }
                    for r in s.itertuples()
                ]
    add_reference_line(fig, y=C.STEP_STABLE_PEARSON, label=f"{C.STEP_STABLE_PEARSON} threshold", row="all", col="all")
    for ci in (1, 2):
        fig.update_xaxes(title_text="finite-difference step (model input units, log)", type="log",
                         row=len(anchors), col=ci)
        for ri in range(1, len(anchors) + 1):
            fig.update_yaxes(title_text="Pearson, adjacent steps", range=[0.9, 1.001], row=ri, col=ci)
    apply_theme(fig, height=760)
    return Fig(
        name="step_stability",
        figure=fig,
        data=rows,
        alt="Adjacent-step agreement of the finite-difference gradient column, by model and anchor.",
        meta={
            "title": "Finite-difference step stability",
            "description": "Each point compares the gradient column computed at that step with the "
            "column at the next larger step, as the median Pearson correlation over 50 perturbed "
            "genes at a 4 CP10K base value. A step is selected only if it also agrees in magnitude "
            "(median relative L2 difference at or below 0.10).",
            "findings": "Where the curve sits on the threshold line the gradient is a property of "
            "the model rather than of the step; where it falls away, the block's gradient features "
            "are usable for direction and ranking but not magnitude.",
            "sources": "experiments/experiment-2-nygwcv/src/scfm_pfm/features/extract.py",
        },
    )


def fig_response_by_value(bundles: dict, out: Path) -> Fig:
    """How large the response difference is at each virtual value."""
    col = colors(out)
    anchors = ["uniform_virtual", "mean_control"]
    fig = _panel_grid(
        len(anchors),
        2,
        [f"{LABEL[cl]}, {ANCHOR_LABEL[a]}" for a in anchors for cl in LINES],
        shared_xaxes=True,
        vertical_spacing=0.12,
    )
    rows = []
    for ri, anchor in enumerate(anchors, start=1):
        for ci, cl in enumerate(LINES, start=1):
            for model in MODELS:
                b = bundles.get((model, cl))
                if not b or anchor not in b.get("query_summary", {}):
                    continue
                s = pd.DataFrame(b["query_summary"][anchor]).sort_values("value_cp10k")
                fig.add_trace(
                    go.Scatter(
                        # the grid is roughly geometric and includes 0, so it reads as
                        # ordered categories rather than on a linear or log axis
                        x=[f"{v:g}" for v in s.value_cp10k],
                        y=s.cross_gene_sd,
                        name=MODEL_LABEL[model] if (ri == 1 and ci == 1) else None,
                        legendgroup=model,
                        showlegend=ri == 1 and ci == 1,
                        mode="lines+markers",
                        line=dict(color=col[MODEL_LABEL[model]]),
                        marker=dict(color=col[MODEL_LABEL[model]]),
                        hovertemplate="value %{x} CP10K<br>spread %{y:.3g}<extra></extra>",
                    ),
                    row=ri,
                    col=ci,
                )
                rows += [
                    {
                        "model": model,
                        "cell_line": cl,
                        "anchor": anchor,
                        "value_cp10k": float(r.value_cp10k),
                        "cross_gene_sd": float(r.cross_gene_sd),
                        "cross_gene_mean": float(r.cross_gene_mean),
                        "max_abs": float(r.max_abs),
                    }
                    for r in s.itertuples()
                ]
    for ci in (1, 2):
        fig.update_xaxes(title_text="virtual value for the perturbed gene (CP10K)",
                         type="category", row=len(anchors), col=ci)
        for ri in range(1, len(anchors) + 1):
            fig.update_xaxes(type="category", row=ri, col=ci)
            fig.update_yaxes(title_text="response spread (log, model units)", type="log",
                             dtick=1, row=ri, col=ci)
    apply_theme(fig, height=620)
    return Fig(
        name="response_by_value",
        figure=fig,
        data=rows,
        alt="Response spread across read-out genes at each virtual value, per model and anchor.",
        meta={
            "title": "Size of the response at each virtual value",
            "description": "Median across queries of the standard deviation of the response "
            "difference over read-out genes, in each model's own output units, which are not "
            "comparable between models (log axis).",
            "findings": "The four scales differ by orders of magnitude, which is why every block "
            "also ships a per-query standardized form.",
            "sources": "experiments/experiment-2-nygwcv/results/report/bundles",
        },
    )


def fig_softmax_coupling(bundles: dict, out: Path) -> Fig:
    """The part of every softmax response that is redistribution."""
    col = colors(out)
    fig = _panel_grid(1, 2, [LABEL[cl] for cl in LINES])
    rows = []
    for ci, cl in enumerate(LINES, start=1):
        for model in MODELS:
            b = bundles.get((model, cl))
            if not b:
                continue
            vals, ratios = [], []
            for anchor in ("uniform_virtual", "mean_control"):
                for rec in b.get("query_summary", {}).get(anchor, []):
                    sd = max(abs(rec["cross_gene_sd"]), 1e-30)
                    vals.append(f"{ANCHOR_LABEL[anchor]}, {rec['value_cp10k']:g}")
                    ratios.append(abs(rec["cross_gene_mean"]) / sd)
            if not vals:
                continue
            fig.add_trace(
                go.Bar(
                    x=vals,
                    y=ratios,
                    name=MODEL_LABEL[model] if ci == 1 else None,
                    legendgroup=model,
                    showlegend=ci == 1,
                    marker_color=col[MODEL_LABEL[model]],
                    hovertemplate="%{x}<br>|mean| / sd %{y:.3g}<extra></extra>",
                ),
                row=1,
                col=ci,
            )
            rows += [
                {"model": model, "cell_line": cl, "query": v, "abs_mean_over_sd": float(r)}
                for v, r in zip(vals, ratios)
            ]
    add_reference_line(fig, y=0.05, label="0.05 tolerance", row="all", col="all")
    for ci in (1, 2):
        fig.update_xaxes(title_text="anchor and virtual value (CP10K)", row=1, col=ci)
        fig.update_yaxes(title_text="|cross-gene mean| / spread (log)", type="log", row=1, col=ci)
    apply_theme(fig, height=460)
    return Fig(
        name="softmax_coupling",
        figure=fig,
        data=rows,
        alt="Cross-gene mean of the response relative to its spread, per model and query.",
        meta={
            "title": "How much of each response is redistribution",
            "description": "For a model whose head ends in a softmax across the supplied genes, "
            "raising one gene's share must lower the others, so the response differences sum to "
            "about zero by construction. The bars show the median cross-gene mean divided by the "
            "cross-gene spread (log axis).",
            "findings": "scPRINT sits below the tolerance at every query, which is the signature "
            "of exact redistribution. TranscriptFormer sits below it at the low values and rises "
            "above it at the top of the grid: its read-out set is delivered in two separately "
            "normalized passes, so the shares sum to one inside a pass and not across the union. "
            "AIDO.Cell sits in the same range without a softmax head, because one gene's query "
            "barely moves a 19,264-gene prediction on average, so a small ratio is not by itself "
            "evidence of redistribution. scGPT, whose output is a per-cell bin index, is the only "
            "one clearly above.",
            "sources": "experiments/experiment-2-nygwcv/results/report/bundles",
        },
    )


def load_hidden(out: Path) -> dict:
    """The visible / masked / median-substituted records, where they exist."""
    hidden: dict[tuple[str, str], dict] = {}
    for d in (paths.repo_dir() / "results" / "report" / "hidden_readout",
              out / "report" / "hidden_readout"):
        if not d.exists():
            continue
        for path in sorted(d.glob("*.json")):
            if path.name.startswith("verification_"):
                continue
            rec = json.loads(path.read_text())
            hidden[(rec["model"], rec["cell_line"])] = rec
    return hidden


#: the three ways the read-out gene's own input value is delivered
HIDDEN_SERIES = (
    ("visible", "visible", "HepG2"),
    ("masked", "mask token", "Jurkat"),
    ("median_substituted", "median level", "S3"),
)


def fig_masked_regime(bundles: dict, out: Path) -> Fig:
    """Three ways of hiding the read-out gene's own value, at the same anchor."""
    col = colors(out)
    hidden = load_hidden(out)
    if not hidden:
        raise RuntimeError("no hidden-readout records to plot")
    fig = _panel_grid(1, 2, [LABEL[cl] for cl in LINES])
    rows = []
    for ci, cl in enumerate(LINES, start=1):
        names = [MODEL_LABEL[m] for m in MODELS if (m, cl) in hidden]
        for key, label, colour in HIDDEN_SERIES:
            values = [
                hidden[(m, cl)]["summary"][key]["abs_median"]
                for m in MODELS
                if (m, cl) in hidden
            ]
            fig.add_trace(
                go.Bar(
                    x=names, y=values, name=label if ci == 1 else None, legendgroup=label,
                    showlegend=ci == 1, marker_color=col[colour],
                    hovertemplate="%{x}<br>" + label + " %{y:.3g}<extra></extra>",
                ),
                row=1, col=ci,
            )
        for m in MODELS:
            rec = hidden.get((m, cl))
            if not rec:
                continue
            rows.append(
                {
                    "model": m,
                    "cell_line": cl,
                    "n_perturbed": rec["n_perturbed"],
                    "substitute_cp10k": rec["substitute_level"]["value_cp10k"],
                    **{
                        f"{key}_abs_median": rec["summary"][key]["abs_median"]
                        for key, _, _ in HIDDEN_SERIES
                    },
                    "masked_over_visible": rec["ratios"].get("masked_over_visible"),
                    "median_substituted_over_visible": rec["ratios"].get(
                        "median_substituted_over_visible"
                    ),
                    "run2_diluted_masked_abs_median": rec["summary"]["masked"][
                        "abs_median_run2_diluted_estimator"
                    ],
                }
            )
    for ci in (1, 2):
        fig.update_xaxes(title_text="model", row=1, col=ci)
        fig.update_yaxes(title_text="median |zero-out response| (log, model units)", type="log",
                         dtick=1, row=1, col=ci)
    apply_theme(fig, height=460)

    survives, artefact = [], []
    for r in rows:
        m, s = r["masked_over_visible"], r["median_substituted_over_visible"]
        if not (m and s):
            continue
        (survives if _same_direction(m, s) else artefact).append(MODEL_LABEL[r["model"]])
    findings = (
        "The mask token and an uninformative in-distribution value are not the same query. "
    )
    if survives:
        findings += (
            f"For {', '.join(sorted(set(survives)))} the substitution moves the response the "
            "same way as the mask, so what the mask removes is information rather than "
            "distribution. "
        )
    if artefact:
        findings += (
            f"For {', '.join(sorted(set(artefact)))} the substitution does not reproduce what "
            "the mask does, so that model's masked response says more about the mask token "
            "than about reconstruction. "
        )
    findings += "TranscriptFormer is absent: it is autoregressive, so a gene never sees its own value."
    return Fig(
        name="masked_regime",
        figure=fig,
        data=rows,
        alt="Zero-out response magnitude with the read-out gene's own value visible, masked, "
        "and replaced by the median control level.",
        meta={
            "title": "Three ways to hide the read-out gene's own value, at the same anchor",
            "description": "Median absolute zero-out response at the mean control anchor over the "
            "same perturbed-gene subset, with the read-out gene's own input value visible, "
            "replaced by each model's own mask convention, and replaced by one uninformative "
            "in-distribution level, the median pooled-control level of the read-out genes (log "
            "axis, each model in its own output units). All three come from one job with one "
            "batching, and all three drop the passes in which the mask would have overwritten "
            "the perturbation itself, so they sit slightly above the shipped masked block's own "
            "numbers.",
            "findings": findings,
            "sources": "experiments/experiment-2-nygwcv/results/report/hidden_readout",
        },
    )


def _same_direction(masked_ratio: float, substituted_ratio: float, share: float = 0.5) -> bool:
    """Does the substitution reproduce the mask's effect, and at least half of it?

    Both ratios are read on a log scale, because a factor of 0.1 and a factor of 10
    are the same size of effect in opposite directions. The substitution counts as
    reproducing the mask when it moves the response the same way and by at least
    ``share`` of the mask's log magnitude.
    """
    import math

    lm, ls = math.log(masked_ratio), math.log(substituted_ratio)
    return lm * ls > 0 and abs(ls) >= share * abs(lm)


#: the three alignment statistics, each read against the same query with a randomly
#: relabelled read-out column: (label, observed key, null key, colour key)
ALIGN_STATS = (
    ("own entry largest", "own_largest", "own_largest_null", "ceiling"),
    ("own entry high in the vector", "_rank_obs", "_rank_null", "mean-effect baseline"),
    ("own output moves as expected", "sign_rate", "sign_rate_null", "cross-perturbation"),
)


def fig_wiring(bundles: dict, out: Path) -> Fig:
    """Three readings of the alignment question, each against its permuted null."""
    col = colors(out)
    fig = _panel_grid(1, 2, [LABEL[cl] for cl in LINES])
    rows = []
    for ci, cl in enumerate(LINES, start=1):
        names, obs, null = [], {k: [] for _, k, _, _ in ALIGN_STATS}, {k: [] for _, _, k, _ in ALIGN_STATS}
        for model in MODELS:
            b = bundles.get((model, cl))
            align = {r["anchor"]: r for r in (b or {}).get("alignment", [])}
            r = align.get("mean_control")
            if not r:
                continue
            names.append(MODEL_LABEL[model])
            # a rank near the top of the vector reads as a high number, like the others
            r = dict(r)
            r["_rank_obs"] = 1.0 - r["self_rank_median_fraction"]
            r["_rank_null"] = 1.0 - r["self_rank_median_fraction_null"]
            for _, ok, nk, _ in ALIGN_STATS:
                obs[ok].append(r[ok])
                null[nk].append(r[nk])
            rows += [
                {"model": model, "cell_line": cl, **a}
                for a in (b or {}).get("alignment", [])
            ]
        for label, ok, nk, ck in ALIGN_STATS:
            fig.add_trace(
                go.Bar(x=names, y=obs[ok], name=label, legendgroup=label, showlegend=ci == 1,
                       marker=dict(color=col[ck]),
                       hovertemplate="%{x}<br>" + label + " %{y:.3f}<extra></extra>"),
                row=1, col=ci,
            )
            fig.add_trace(
                go.Bar(x=names, y=null[nk], name=label + ", permuted", legendgroup=label,
                       showlegend=ci == 1,
                       marker=dict(color=lighten(col[ck]),
                                   pattern=dict(shape="/", fgcolor=col[ck], size=4, solidity=0.25),
                                   line=dict(color=col[ck], width=1)),
                       hovertemplate="%{x}<br>" + label + ", permuted %{y:.3f}<extra></extra>"),
                row=1, col=ci,
            )
    add_reference_line(fig, y=C.SELF_RESPONSE_MIN_PASS, label="0.95 threshold", row="all", col="all")
    for ci in (1, 2):
        fig.update_xaxes(title_text="model", row=1, col=ci)
        fig.update_yaxes(title_text="statistic (1 = perfectly aligned)", range=[0, 1.05],
                         row=1, col=ci)
    apply_theme(fig, height=470)
    return Fig(
        name="wiring_checks",
        figure=fig,
        data=rows,
        alt="Three alignment statistics per model beside their permuted nulls; every model "
        "separates from its null on at least one of them.",
        meta={
            "title": "Read-out order alignment, each statistic against its permuted null",
            "description": "At the mean control anchor, over the perturbed genes sampled in each "
            "block. Each statistic is shown beside the same statistic computed after relabelling "
            "the perturbed gene's own read-out column at random (the pale bar). 'Own entry "
            "largest' is the pre-registered guard; 'own entry high in the vector' is one minus "
            "the median rank of the own entry as a fraction of the vector; 'own output moves as "
            "expected' is the sign reading.",
            "findings": "scPRINT saturates the pre-registered guard on both lines. scGPT and "
            "AIDO.Cell miss its 0.95 threshold while the own entry still ranks first for the "
            "median gene, so the rank statistic is what establishes their alignment. "
            "TranscriptFormer scores exactly zero on that guard, and so does its null: with the "
            "perturbed gene placed first, an autoregressive model never conditions on its own "
            "value, so the guard cannot have power there. Every block separates from its null on "
            "at least one statistic at every anchor.",
            "sources": "experiments/experiment-2-nygwcv/results/report/bundles",
        },
    )


def fig_scgpt_context(bundles: dict, out: Path) -> Fig:
    """How far scGPT's responses move between supplied-set widths."""
    col = colors(out)
    fig = _panel_grid(1, 2, ["Direction (Pearson)", "Magnitude (relative L2 difference)"])
    rows = []
    for cl in LINES:
        b = bundles.get(("scgpt", cl))
        cc = (b or {}).get("manifest", {}).get("context_comparison")
        if not cc:
            continue
        pairs = list(cc["agreement"])
        labels = [p.replace("_vs_", " vs ").replace("_", " ") for p in pairs]
        fig.add_trace(
            go.Bar(x=labels, y=[cc["agreement"][p]["median_pearson"] for p in pairs],
                   name=LABEL[cl], legendgroup=cl, showlegend=True,
                   marker_color=col[LABEL[cl]],
                   hovertemplate="%{x}<br>Pearson %{y:.4f}<extra></extra>"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=labels, y=[cc["agreement"][p]["median_relative_l2"] for p in pairs],
                   name=None, legendgroup=cl, showlegend=False,
                   marker_color=col[LABEL[cl]],
                   hovertemplate="%{x}<br>relative L2 %{y:.4f}<extra></extra>"),
            row=1, col=2,
        )
        rows += [
            {"cell_line": cl, "pair": p, **cc["agreement"][p], "widths": cc["widths"]}
            for p in pairs
        ]
    fig.update_xaxes(title_text="supplied-set widths compared", row=1, col=1)
    fig.update_xaxes(title_text="supplied-set widths compared", row=1, col=2)
    fig.update_yaxes(title_text="median Pearson", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="median relative L2 difference", row=1, col=2)
    apply_theme(fig, height=460)
    return Fig(
        name="scgpt_context",
        figure=fig,
        data=rows,
        alt="scGPT responses agree in direction but not magnitude across supplied-set widths.",
        meta={
            "title": "scGPT: what the supplied-set width changes",
            "description": "Zero-out response differences for the same perturbed genes at the mean "
            "control anchor, computed with three supplied widths: the expressed read-out set, run "
            "1's full 7,444-gene universe, and tiles of at most 1,200 genes sharing a fixed prefix. "
            "A relative L2 difference of 0 means identical vectors, 2 means opposite.",
            "findings": "Directions largely agree and magnitudes do not, so the width is a "
            "measurement choice rather than a detail; the production sweep uses the tiled width, "
            "which is the one inside scGPT's training length.",
            "sources": "experiments/experiment-2-nygwcv/results/report/bundles",
        },
    )


def fig_feature_scale(bundles: dict, out: Path) -> Fig:
    """The four output scales, per feature form."""
    col = colors(out)
    forms = ("zero_out", "vvp", "gdt")
    fig = _panel_grid(len(forms), 1, [f"{f.replace('_', ' ')}" for f in forms],
                      shared_xaxes=True, vertical_spacing=0.1)
    rows = []
    for ri, form in enumerate(forms, start=1):
        for model in MODELS:
            xs, ys, texts = [], [], []
            for cl in LINES:
                b = bundles.get((model, cl))
                blk = (b or {}).get("blocks", {}).get(f"mean_control/{form}")
                if not blk:
                    continue
                xs += [blk["abs_median"], blk["abs_p99"]]
                ys += [MODEL_LABEL[model], MODEL_LABEL[model]]
                texts += [f"{LABEL[cl]} median", f"{LABEL[cl]} 99th percentile"]
                rows.append({"model": model, "cell_line": cl, "form": form,
                             "abs_median": blk["abs_median"], "abs_p99": blk["abs_p99"]})
            if xs:
                fig.add_trace(
                    go.Scatter(x=xs, y=ys, mode="markers",
                               name=MODEL_LABEL[model] if ri == 1 else None,
                               legendgroup=model, showlegend=ri == 1,
                               marker=dict(color=col[MODEL_LABEL[model]], size=9,
                                           symbol=["circle", "diamond"] * (len(xs) // 2)),
                               text=texts,
                               hovertemplate="%{text}<br>|value| %{x:.3g}<extra></extra>"),
                    row=ri, col=1,
                )
    fig.update_xaxes(title_text="absolute feature value (log, model units)", type="log",
                     dtick=1, row=len(forms), col=1)
    for ri in range(1, len(forms) + 1):
        fig.update_yaxes(title_text="", row=ri, col=1)
        # decade ticks only; the axis title (and its "log scale" note) stays on the last row
        fig.update_xaxes(dtick=1, row=ri, col=1)
    apply_theme(fig, height=700)
    return Fig(
        name="feature_scale",
        figure=fig,
        data=rows,
        alt="Feature magnitudes span orders of magnitude across the four models.",
        meta={
            "title": "The four output scales, per feature form",
            "description": "Median and 99th percentile of the absolute raw feature value at the "
            "mean control anchor, in each model's own output units (log axis). Circles are medians, "
            "diamonds 99th percentiles; both cell lines are shown.",
            "findings": "Raw features are not comparable across models, which is what the "
            "standardized and, where defensible, predicted log fold-change forms are for.",
            "sources": "experiments/experiment-2-nygwcv/results/report/bundles",
        },
    )


BUILDERS_FLOOR = {
    "floor_ceiling": fig_floor_ceiling,
    "floor_shared_response": fig_floor_shared_response,
}
BUILDERS_BLOCK = {
    "step_stability": fig_step_stability,
    "response_by_value": fig_response_by_value,
    "softmax_coupling": fig_softmax_coupling,
    "masked_regime": fig_masked_regime,
    "wiring_checks": fig_wiring,
    "scgpt_context": fig_scgpt_context,
    "feature_scale": fig_feature_scale,
}


def build_all(bundles: dict, out: Path) -> dict[str, Fig]:
    floor = paths.repo_dir() / "results" / "floor_sweep"
    summary = pd.read_parquet(floor / "floor_ceiling_summary.parquet")
    shared = pd.read_parquet(floor / "floor_shared_response.parquet")
    figs: dict[str, Fig] = {}
    for name, fn in BUILDERS_FLOOR.items():
        figs[name] = fn(summary if name == "floor_ceiling" else shared, out)
    for name, fn in BUILDERS_BLOCK.items():
        try:
            figs[name] = fn(bundles, out)
        except Exception as exc:  # noqa: BLE001
            print(f"[figures] {name} skipped: {type(exc).__name__}: {exc}", flush=True)
    for name, fig in figs.items():
        fig.save(out)
        print(f"[figures] saved {name}", flush=True)
    return figs
