"""Figures, built pod-side from the summary tables the sweep produced.

Every figure reads a small table under ``results/summary`` and saves a bundle
under ``figures/<name>/`` so it stays re-renderable from its own data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle

from .. import paths

PRIMARY = "residual__weighted__disattenuated"

FAMILY_ORDER = ["scfm", "hybrid", "statistical", "node_only", "baseline", "null", "scfm_anchor"]
FAMILY_LABEL = {
    "scfm": "foundation model",
    "hybrid": "hybrid",
    "statistical": "co-expression",
    "node_only": "node only",
    "baseline": "mean effect",
    "null": "null",
    "scfm_anchor": "other anchor",
}


def _line_label(line: str) -> str:
    return {"hepg2": "HepG2", "jurkat": "Jurkat"}.get(line, line)


def arm_scores(summary_dir: Path, out_root: Path, translator: str = "mlp") -> dict:
    """Dot plot of the pre-registered score per arm, one panel per cell line."""
    runs = pd.read_parquet(summary_dir / "runs.parquet")
    d = runs[(runs.scope == "full") & (runs.split == "grouped") & (runs.translator == translator)]
    lines = sorted(d.cell_line.unique())
    fig = make_subplots(
        rows=1, cols=len(lines), shared_xaxes=False,
        subplot_titles=[_line_label(x) for x in lines], horizontal_spacing=0.16,
    )
    payload = []
    for k, line in enumerate(lines, start=1):
        block = d[d.cell_line == line].sort_values(PRIMARY)
        for fam in FAMILY_ORDER:
            sub = block[block.family == fam]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub[PRIMARY],
                    y=sub.arm,
                    mode="markers",
                    name=FAMILY_LABEL[fam],
                    legendgroup=fam,
                    showlegend=k == 1,
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=sub[PRIMARY + "_hi"] - sub[PRIMARY],
                        arrayminus=sub[PRIMARY] - sub[PRIMARY + "_lo"],
                        thickness=1.2,
                        width=0,
                    ),
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                ),
                row=1, col=k,
            )
        add_reference_line(fig, x=0.0, label="shuffled-label null", row=1, col=k)
        add_reference_line(fig, x=1.0, label="reproducible ceiling", row=1, col=k)
        payload += block.assign(cell_line=line)[
            ["cell_line", "arm", "family", "feature_dim", PRIMARY, PRIMARY + "_lo", PRIMARY + "_hi"]
        ].to_dict("records")
        fig.update_xaxes(title_text="disattenuated residual correlation", row=1, col=k)
    fig.update_yaxes(title_text="", automargin=True)
    apply_theme(fig, height=560)
    save_figure_bundle(
        fig,
        "arm_scores",
        data={"records": payload, "metric": PRIMARY, "translator": translator},
        alt=(
            "Dot plot of the disattenuated residual correlation for every arm, HepG2 and "
            "Jurkat panels, with bootstrap intervals; 0 marks the shuffled-label null and "
            "1.0 the reproducible ceiling."
        ),
        root=out_root,
    )
    return {"figure": "arm_scores", "n_records": len(payload)}


def vs_coexpression(summary_dir: Path, out_root: Path, translator: str = "mlp") -> dict:
    """Paired difference against the co-expression arm, with the achieved
    resolution drawn as a band."""
    con = pd.read_parquet(summary_dir / "contrasts.parquet")
    d = con[(con.scope == "full") & (con.split == "grouped") & (con.translator == translator)]
    lines = sorted(d.cell_line.unique())
    fig = make_subplots(
        rows=1, cols=len(lines), subplot_titles=[_line_label(x) for x in lines],
        horizontal_spacing=0.16,
    )
    payload = []
    for k, line in enumerate(lines, start=1):
        block = d[d.cell_line == line].sort_values("median_difference")
        fig.add_trace(
            go.Scatter(
                x=block.median_difference,
                y=block.arm,
                mode="markers",
                name="difference",
                showlegend=False,
                error_x=dict(
                    type="data", symmetric=False,
                    array=block.hi - block.median_difference,
                    arrayminus=block.median_difference - block.lo,
                    thickness=1.2, width=0,
                ),
                hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
            ),
            row=1, col=k,
        )
        add_reference_line(fig, x=0.0, label="co-expression", row=1, col=k)
        mrd = float(np.nanmedian(block.min_resolvable_difference_achieved))
        add_reference_line(fig, x=mrd, label=f"resolvable {mrd:.3f}", row=1, col=k)
        fig.update_xaxes(title_text="difference vs co-expression", row=1, col=k)
        payload += block.assign(cell_line=line).to_dict("records")
    fig.update_yaxes(title_text="", automargin=True)
    apply_theme(fig, height=520)
    save_figure_bundle(
        fig,
        "vs_coexpression",
        data={"records": payload, "translator": translator},
        alt=(
            "Paired difference between each arm and the co-expression arm on the same "
            "held-out perturbations, with bootstrap intervals and the smallest difference "
            "the run can resolve."
        ),
        root=out_root,
    )
    return {"figure": "vs_coexpression", "n_records": len(payload)}


def grid_cells(summary_dir: Path, out_root: Path, arms: list[str] | None = None) -> dict:
    """The whole correlation grid for the arms that carry the claim."""
    grid = pd.read_parquet(summary_dir / "grid.parquet")
    d = grid[(grid.scope == "full") & (grid.split == "grouped") & (grid.translator == "mlp")]
    if arms:
        d = d[d.arm.isin(arms)]
    d = d.assign(cell=d.target + " / " + d.weighting + " / " + d.correction)
    lines = sorted(d.cell_line.unique())
    fig = make_subplots(
        rows=1, cols=len(lines), subplot_titles=[_line_label(x) for x in lines],
        horizontal_spacing=0.16,
    )
    for k, line in enumerate(lines, start=1):
        block = d[d.cell_line == line]
        pivot = block.pivot_table(index="arm", columns="cell", values="value")
        fig.add_trace(
            go.Heatmap(
                z=pivot.to_numpy(),
                x=list(pivot.columns),
                y=list(pivot.index),
                coloraxis="coloraxis",
                hovertemplate="%{y}<br>%{x}: %{z:.4f}<extra></extra>",
            ),
            row=1, col=k,
        )
        fig.update_xaxes(title_text="target / weighting / correction", row=1, col=k, tickangle=40)
    fig.update_yaxes(title_text="", automargin=True)
    apply_theme(fig, height=520)
    save_figure_bundle(
        fig,
        "grid_cells",
        data={"records": d.to_dict("records")},
        alt=(
            "Heatmap of all eight cells of the correlation grid for the main arms: the "
            "residual and raw targets, each weighted and unweighted, each with and without "
            "the ceiling correction."
        ),
        root=out_root,
    )
    return {"figure": "grid_cells", "n_records": int(d.shape[0])}


def split_gap(summary_dir: Path, out_root: Path) -> dict:
    """Grouped versus random split: how much of a score is family memorization."""
    runs = pd.read_parquet(summary_dir / "runs.parquet")
    d = runs[(runs.scope == "full") & (runs.translator == "mlp")]
    piv = d.pivot_table(index=["cell_line", "arm", "family"], columns="split", values=PRIMARY)
    piv = piv.dropna().reset_index()
    fig = go.Figure()
    for fam in FAMILY_ORDER:
        sub = piv[piv.family == fam]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["grouped"], y=sub["random"], mode="markers", name=FAMILY_LABEL[fam],
                text=sub.cell_line + " · " + sub.arm,
                hovertemplate="%{text}<br>grouped %{x:.4f}, random %{y:.4f}<extra></extra>",
            )
        )
    lim = [float(min(piv["grouped"].min(), piv["random"].min())) - 0.02,
           float(max(piv["grouped"].max(), piv["random"].max())) + 0.02]
    fig.add_trace(
        go.Scatter(x=lim, y=lim, mode="lines", name="equal", line=dict(dash="dot", width=1),
                   hoverinfo="skip")
    )
    fig.update_xaxes(title_text="grouped split (whole response families held out)")
    fig.update_yaxes(title_text="random split")
    apply_theme(fig, height=460)
    save_figure_bundle(
        fig,
        "split_gap",
        data={"records": piv.to_dict("records")},
        alt=(
            "Each arm's score under the grouped split against the random split; points above "
            "the diagonal score better when related perturbed genes can appear in both sides."
        ),
        root=out_root,
    )
    return {"figure": "split_gap", "n_records": int(piv.shape[0])}


def transfer_scores(summary_dir: Path, out_root: Path) -> dict | None:
    path = summary_dir / "transfer_runs.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    d = d[d.translator == "mlp"]
    fig = go.Figure()
    for fam in FAMILY_ORDER:
        sub = d[d.family == fam].sort_values(PRIMARY)
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub[PRIMARY], y=sub.arm + " · " + sub.cell_line, mode="markers",
                name=FAMILY_LABEL[fam],
                error_x=dict(
                    type="data", symmetric=False,
                    array=sub[PRIMARY + "_hi"] - sub[PRIMARY],
                    arrayminus=sub[PRIMARY] - sub[PRIMARY + "_lo"], thickness=1.2, width=0,
                ),
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            )
        )
    add_reference_line(fig, x=0.0, label="no transfer")
    fig.update_xaxes(title_text="disattenuated residual correlation on the held-out cell line")
    fig.update_yaxes(title_text="", automargin=True)
    apply_theme(fig, height=520)
    save_figure_bundle(
        fig,
        "transfer",
        data={"records": d.to_dict("records")},
        alt="Cross-cell-line transfer: trained on one screen, scored on the other.",
        root=out_root,
    )
    return {"figure": "transfer", "n_records": int(d.shape[0])}


def discrimination_vs_correlation(summary_dir: Path, out_root: Path) -> dict:
    runs = pd.read_parquet(summary_dir / "runs.parquet")
    d = runs[(runs.scope == "full") & (runs.split == "grouped") & (runs.translator == "mlp")]
    fig = go.Figure()
    for fam in FAMILY_ORDER:
        sub = d[d.family == fam]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub[PRIMARY], y=sub.discrimination_ours_residual_mean, mode="markers",
                name=FAMILY_LABEL[fam], text=sub.cell_line + " · " + sub.arm,
                hovertemplate="%{text}<br>correlation %{x:.4f}, discrimination %{y:.4f}"
                              "<extra></extra>",
            )
        )
    add_reference_line(fig, y=0.5, label="chance discrimination")
    fig.update_xaxes(title_text="disattenuated residual correlation")
    fig.update_yaxes(title_text="perturbation discrimination (residual vectors)")
    apply_theme(fig, height=460)
    save_figure_bundle(
        fig,
        "discrimination_vs_correlation",
        data={"records": d[
            ["cell_line", "arm", "family", PRIMARY, "discrimination_ours_residual_mean"]
        ].to_dict("records")},
        alt=(
            "Two scoring axes against each other: the disattenuated residual correlation and "
            "whether a prediction is closer to its own perturbation than to the others."
        ),
        root=out_root,
    )
    return {"figure": "discrimination_vs_correlation", "n_records": int(d.shape[0])}


def reference_quantities(bundle_manifest: dict, out_root: Path) -> dict:
    """What is reproducible at all, per cell line, and what the mean-effect
    baseline already reaches."""
    rows = []
    for line, block in bundle_manifest["lines"].items():
        ref = block["reference"]
        rows.append(
            {
                "cell_line": _line_label(line),
                "residual ceiling": ref["residual_ceiling"]["median_spearman_brown"],
                "raw-effect ceiling": ref["raw_effect_ceiling"]["median_spearman_brown"],
                "median gene reliability": ref["reliability"]["median_weight"],
                "residual share of variance": ref["shared_response"][
                    "median_residual_fraction_of_variance"
                ],
            }
        )
    frame = pd.DataFrame(rows)
    fig = go.Figure()
    for col in frame.columns[1:]:
        fig.add_trace(go.Bar(x=frame.cell_line, y=frame[col], name=col))
    fig.update_yaxes(title_text="value")
    fig.update_xaxes(title_text="cell line")
    apply_theme(fig, height=420)
    save_figure_bundle(
        fig,
        "reference_quantities",
        data={"records": frame.to_dict("records")},
        alt=(
            "Reference quantities per cell line: how much of the residual and raw response is "
            "reproducible, the median per-gene reliability, and the residual share of variance."
        ),
        root=out_root,
    )
    return {"figure": "reference_quantities", "n_records": int(frame.shape[0])}


def build_all(summary_dir: Path, manifest: dict, out_root: Path | None = None) -> list[dict]:
    out_root = out_root or paths.repo_dir()
    made = [
        reference_quantities(manifest, out_root),
        arm_scores(summary_dir, out_root),
        vs_coexpression(summary_dir, out_root),
        grid_cells(summary_dir, out_root),
        split_gap(summary_dir, out_root),
        discrimination_vs_correlation(summary_dir, out_root),
    ]
    t = transfer_scores(summary_dir, out_root)
    if t:
        made.append(t)
    return [m for m in made if m]
