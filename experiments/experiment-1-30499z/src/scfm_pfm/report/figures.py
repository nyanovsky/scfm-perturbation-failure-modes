"""Figures for the data-preparation report.

Every figure reads a file that a later run can read too, so nothing here is a
one-off rendering of numbers that live only in this script. Run pod-side against
a mirror of the artifact tree:

    uv run python -m scfm_pfm.report.figures --root results/artifacts_mirror
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import PLOTLY_CONFIG, add_reference_line, apply_theme, save_figure_bundle

from .. import config as C
from .. import paths

LINES = ("hepg2", "jurkat")
LABEL = {"hepg2": "HepG2", "jurkat": "Jurkat"}
STRATA = list(C.STRATUM_NAMES)


def _qc(root: Path, cl: str) -> dict:
    return json.loads((root / "effects" / cl / "qc.json").read_text())


def fig_funnel(root: Path, out: Path) -> dict:
    """Where the cells and perturbations go, step by step."""
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
                hovertemplate="%{x}<br>%{y:,} cells<extra>" + LABEL[cl] + "</extra>",
            )
        )
    fig.update_xaxes(title_text="filter step, in order")
    fig.update_yaxes(title_text="cells remaining")
    apply_theme(fig, height=430)
    save_figure_bundle(
        fig,
        "filter_funnel",
        data=df.to_dict("records"),
        alt=(
            "Cells remaining after each filter step in both cell lines. Per-cell QC "
            "removes very little; the working set is set by the perturbation-level "
            "thresholds."
        ),
        root=out,
    )
    return {"rows": len(df)}


def fig_knockdown(root: Path, out: Path) -> dict:
    """On-target knockdown, the check that the perturbation did what it claims."""
    frames = []
    for cl in LINES:
        p = pd.read_parquet(root / "effects" / cl / "perturbations.parquet")
        p = p.assign(cell_line=LABEL[cl])
        frames.append(p)
    df = pd.concat(frames, ignore_index=True)
    fig = go.Figure()
    for cl in LINES:
        sub = df[(df.cell_line == LABEL[cl]) & df.knockdown.notna()]
        fig.add_trace(
            go.Histogram(
                x=sub.knockdown,
                name=f"{LABEL[cl]} (n={sub.shape[0]:,})",
                opacity=0.65,
                xbins={"start": -0.5, "end": 1.0, "size": 0.02},
            )
        )
    fig.update_layout(barmode="overlay")
    add_reference_line(
        fig, x=C.EFFECTS.knockdown_detect_min,
        label=f"detection floor {C.EFFECTS.knockdown_detect_min:.2f}",
    )
    fig.update_xaxes(title_text="on-target knockdown, 1 - (perturbed / control) CP10K")
    fig.update_yaxes(title_text="perturbations")
    apply_theme(fig, height=420)
    medians = {
        LABEL[cl]: float(df[df.cell_line == LABEL[cl]].knockdown.median()) for cl in LINES
    }
    save_figure_bundle(
        fig,
        "knockdown_distribution",
        data={
            "medians": medians,
            "values": {
                LABEL[cl]: df[df.cell_line == LABEL[cl]].knockdown.dropna().round(4).tolist()
                for cl in LINES
            },
        },
        alt=(
            "Distribution of on-target knockdown per perturbation in both cell lines, "
            f"medians {medians}."
        ),
        root=out,
    )
    return medians


def fig_effect_size(root: Path, out: Path) -> dict:
    """Effect size against differential expression, coloured by stratum."""
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=[LABEL[cl] for cl in LINES],
        shared_yaxes=True, horizontal_spacing=0.08,
    )
    summary = {}
    for i, cl in enumerate(LINES, start=1):
        p = pd.read_parquet(root / "effects" / cl / "perturbations.parquet")
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
                    marker={"size": 4, "opacity": 0.5},
                    text=sub.target,
                    hovertemplate="%{text}<br>effect L2 %{x:.2f}<br>%{y} DE genes + 1<extra></extra>",
                ),
                row=1, col=i,
            )
        summary[LABEL[cl]] = {
            s: {
                "n": int((p.effect_stratum == s).sum()),
                "median_de": float(p[p.effect_stratum == s].n_de_genes.median()),
                "median_effect_l2": float(p[p.effect_stratum == s].effect_l2_lfc.median()),
            }
            for s in STRATA
        }
        fig.update_xaxes(title_text="effect size, L2 norm of log2FC", row=1, col=i)
    fig.update_yaxes(title_text="differentially expressed genes + 1", type="log", row=1, col=1)
    apply_theme(fig, height=440)
    save_figure_bundle(
        fig,
        "effect_size_vs_de",
        data=summary,
        alt=(
            "Effect size against the number of differentially expressed genes for every "
            "retained perturbation, split by effect-size stratum."
        ),
        root=out,
    )
    return summary


def fig_ceilings(root: Path, out: Path) -> dict:
    """The noise ceiling: how well a perturbation reproduces itself."""
    frames = []
    for cl in LINES:
        c = pd.read_parquet(root / "effects" / cl / "noise_ceiling_summary.parquet")
        frames.append(c.assign(cell_line=LABEL[cl]))
    df = pd.concat(frames, ignore_index=True)
    metrics = [m for m in C.CEILING_METRICS]
    fig = make_subplots(
        rows=1, cols=len(metrics), subplot_titles=metrics, shared_yaxes=True,
        horizontal_spacing=0.06,
    )
    for j, metric in enumerate(metrics, start=1):
        for cl in LINES:
            for split in C.CEILING_SPLITS:
                sub = df[(df.cell_line == LABEL[cl]) & (df.metric == metric) & (df.split == split)]
                sub = sub[sub.stratum.isin(STRATA)].set_index("stratum").reindex(STRATA).reset_index()
                fig.add_trace(
                    go.Bar(
                        x=sub.stratum,
                        y=sub["median"],
                        name=f"{LABEL[cl]}, {split.replace('_', ' ')}",
                        legendgroup=f"{cl}{split}",
                        showlegend=j == 1,
                        hovertemplate="%{x}<br>median %{y:.4f}<extra></extra>",
                    ),
                    row=1, col=j,
                )
        fig.update_xaxes(title_text="effect-size stratum", row=1, col=j)
    fig.update_yaxes(title_text="median replicate agreement", row=1, col=1)
    apply_theme(fig, height=440)
    save_figure_bundle(
        fig,
        "noise_ceilings",
        data=df[df.stratum.isin(STRATA + ["all"])].to_dict("records"),
        alt=(
            "Median split-half agreement of each perturbation with itself, by effect-size "
            "stratum, metric, cell line and split form. This is the ceiling any model "
            "prediction of these effects can reach."
        ),
        root=out,
    )
    return {"rows": int(df.shape[0])}


def fig_crosscheck(root: Path, out: Path) -> dict:
    """Our estimates against the published reference, and against each other."""
    cc = root / "crosscheck"
    summary = json.loads((cc / "summary.json").read_text())
    pairs = pd.read_parquet(cc / "reference_agreement.parquet")
    own = pd.read_parquet(cc / "own_estimator_agreement.parquet")
    scatter_path = cc / "scatter_sample.parquet"
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.1,
        subplot_titles=[
            "per-gene log2FC against the reference",
            "agreement per perturbation",
        ],
    )
    if scatter_path.exists():
        pts = pd.read_parquet(scatter_path)
        fig.add_trace(
            go.Scattergl(
                x=pts.our_log2fc, y=pts.reference_log2fc, mode="markers",
                marker={"size": 3, "opacity": 0.35}, name="genes",
                text=pts.target,
                hovertemplate="%{text}<br>ours %{x:.3f}<br>reference %{y:.3f}<extra></extra>",
            ),
            row=1, col=1,
        )
        lo = float(min(pts.our_log2fc.min(), pts.reference_log2fc.min()))
        hi = float(max(pts.our_log2fc.max(), pts.reference_log2fc.max()))
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                       line={"dash": "dash"}),
            row=1, col=1,
        )
    fig.add_trace(
        go.Histogram(x=pairs.spearman, name="vs reference", opacity=0.7, nbinsx=50),
        row=1, col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=own.pearson_lfc_within_vs_pooled,
            name="batch-matched vs batch-pooled", opacity=0.7, nbinsx=50,
        ),
        row=1, col=2,
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="our log2 fold change", row=1, col=1)
    fig.update_yaxes(title_text="reference log2 fold change", row=1, col=1)
    fig.update_xaxes(title_text="correlation across genes", row=1, col=2)
    fig.update_yaxes(title_text="perturbations", row=1, col=2)
    apply_theme(fig, height=440)
    save_figure_bundle(
        fig,
        "reference_crosscheck",
        data={
            "summary": summary,
            "per_pair_spearman_quantiles": pairs.spearman.quantile(
                [0.05, 0.25, 0.5, 0.75, 0.95]
            ).round(4).to_dict(),
        },
        alt=(
            f"Our per-gene log2 fold changes against the tahoebio reference release "
            f"(pooled Spearman {summary['pooled_spearman']:.4f} over "
            f"{summary['n_pooled_entries']:,} entries), and the distribution of "
            f"per-perturbation agreement."
        ),
        root=out,
    )
    return summary


def fig_model_coverage(root: Path, out: Path) -> dict:
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
    for col in ("vocabulary", "resolved to Ensembl", "covers HepG2 measured", "covers Jurkat measured"):
        fig.add_trace(
            go.Bar(x=df.model, y=df[col], name=col,
                   hovertemplate="%{x}<br>" + col + ": %{y:,}<extra></extra>")
        )
    add_reference_line(
        fig, y=cov["universe_size"],
        label=f"shared universe {cov['universe_size']:,} genes",
    )
    fig.update_xaxes(title_text="model")
    fig.update_yaxes(title_text="genes", type="log")
    apply_theme(fig, height=430)
    save_figure_bundle(
        fig,
        "model_gene_coverage",
        data={"per_model": rows, "universe_size": cov["universe_size"],
              "shared_measured_genes": cov["shared_measured_genes"],
              "identifier_audit": cov["identifier_audit"]},
        alt=(
            "Vocabulary size, genes resolved to a distinct Ensembl id, and coverage of "
            "the measured genes in each cell line, for the four models. The reference "
            "line is the shared universe every model can address."
        ),
        root=out,
    )
    return {"universe_size": cov["universe_size"], "models": len(rows)}


def fig_model_responses(root: Path, out: Path) -> dict:
    """The four response vectors on the same cell, on their own scales."""
    models = [m for m in ("scgpt", "aido_cell", "scprint", "transcriptformer")
              if (root / "models" / m / "response_vector.parquet").exists()]
    if not models:
        return {"models": 0}
    fig = make_subplots(
        rows=1, cols=len(models), subplot_titles=models, horizontal_spacing=0.05
    )
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
                hovertemplate="%{text}<br>log1p(count) %{x:.2f}<br>response %{y:.4g}<extra></extra>",
            ),
            row=1, col=i,
        )
        fig.update_xaxes(title_text="log1p(observed count)", row=1, col=i)
        summary[m] = {
            "spearman_response_vs_input": meta["response_summary"]["spearman_response_vs_input"],
            "median_expressed": float(r.response[expressed].median()),
            "median_unexpressed": float(r.response[~expressed].median()),
            "output_semantics": json.loads((root / "models" / m / "contract.json").read_text())[
                "output_semantics"
            ],
        }
    fig.update_yaxes(title_text="model response (own scale)", row=1, col=1)
    apply_theme(fig, height=420)
    save_figure_bundle(
        fig,
        "model_responses_on_one_cell",
        data=summary,
        alt=(
            "Each model's per-gene response on the same real control cell, against the "
            "observed count of that gene. The vertical scales are not comparable: the "
            "models emit different quantities."
        ),
        root=out,
    )
    return summary


BUILDERS = {
    "filter_funnel": fig_funnel,
    "knockdown_distribution": fig_knockdown,
    "effect_size_vs_de": fig_effect_size,
    "noise_ceilings": fig_ceilings,
    "reference_crosscheck": fig_crosscheck,
    "model_gene_coverage": fig_model_coverage,
    "model_responses_on_one_cell": fig_model_responses,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None, help="directory the figure bundles land in")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else paths.input_root()
    out = Path(args.out) if args.out else paths.repo_dir() / "figures"
    results = {}
    for name, fn in BUILDERS.items():
        if args.only and name not in args.only:
            continue
        try:
            results[name] = fn(root, out)
            print(f"[figure] {name}: ok", flush=True)
        except FileNotFoundError as exc:
            results[name] = {"skipped": str(exc)}
            print(f"[figure] {name}: skipped ({exc})", flush=True)
    print(json.dumps(results, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
