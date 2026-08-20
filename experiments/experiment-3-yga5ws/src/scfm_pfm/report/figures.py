"""Figures, built pod-side from the summary tables the sweep produced.

Every figure reads a small table under ``results/summary`` and saves a bundle
under ``figures/<name>/`` so it stays re-renderable from its own data.

Scores are ranked on the fold-centred cell of the grid, because a
perturbation-invariant prediction already reaches 0.2 on the pre-registered
cell; the pre-registered number travels alongside it in every bundle's data.

Colour is per point rather than per trace wherever one entity appears in more
than one panel: the package's colourblind check compares every pair of series
colours, and two panels showing the same model are not two series a reader can
confuse. The legend is drawn from empty marker traces, one per entity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle

from .. import paths

#: guide lines (equality diagonals, error bars) are drawn in a light neutral that
#: is not one of the entity colours, so nothing in a legend shares its colour.
GUIDE = "#A8A399"

#: the cell the verdict is read from, and the cell config.py pre-registered.
VERDICT = "centered__weighted__disattenuated"
PREREG = "residual__weighted__disattenuated"

SHUFFLE_TAG = "+shuffled_labels_d"

ENTITY_LABEL = {
    "scgpt": "scGPT",
    "scprint": "scPRINT",
    "transcriptformer": "TranscriptFormer",
    "aido_cell": "AIDO.Cell",
    "coexp": "co-expression",
    "node": "node only",
    "mean_only": "mean effect",
    "null": "null",
}
ENTITY_ORDER = list(ENTITY_LABEL)


def entity_colors() -> dict:
    path = paths.repo_dir() / "figures" / "entity_colors.json"
    raw = json.loads(path.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


COLOR = entity_colors()


def entity_of(arm: str) -> str:
    """Which compared thing an arm belongs to."""
    if SHUFFLE_TAG in arm or arm.startswith("null_"):
        return "null"
    for model in ("scgpt", "scprint", "transcriptformer", "aido_cell"):
        if arm.startswith(model):
            return model
    if arm.startswith("coexp"):
        return "coexp"
    if arm.startswith("node"):
        return "node"
    return "mean_only"


def _line_label(line: str) -> str:
    return {"hepg2": "HepG2", "jurkat": "Jurkat"}.get(line, line)


def _direction_label(s: str) -> str:
    return " to ".join(_line_label(x) for x in s.split("->"))


def _with_entity(d: pd.DataFrame) -> pd.DataFrame:
    return d.assign(entity=d.arm.map(entity_of))


def _colors(entities) -> list[str]:
    return [COLOR[e] for e in entities]


def _legend_dummies(fig, entities: set[str], row: int = 1, col: int = 1) -> None:
    """One empty marker trace per entity: the legend, without colour collisions."""
    for ent in ENTITY_ORDER:
        if ent not in entities:
            continue
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers", name=ENTITY_LABEL[ent],
                marker=dict(color=COLOR[ent], size=8), hoverinfo="skip",
            ),
            row=row, col=col,
        )


def _null_ceiling(runs: pd.DataFrame, col: str = VERDICT) -> float:
    d = runs[runs.arm.str.contains(SHUFFLE_TAG, regex=False)]
    return float(np.nanmax(d[col].to_numpy(dtype=float))) if not d.empty else float("nan")


def _primary(runs: pd.DataFrame, split: str = "grouped", scope: str = "full") -> pd.DataFrame:
    return runs[
        (runs.scope == scope)
        & (runs.split == split)
        & (runs.translator == "mlp")
        & (runs.target_source == "run1_all_controls")
    ]


def _dot_grid(
    frame: pd.DataFrame,
    value: str,
    lo: str,
    hi: str,
    panel_col: str,
    panel_titles: dict,
    x_title: str,
    reference: dict,
    height: int,
    ncols: int = 2,
) -> go.Figure:
    """One sorted dot plot per panel, points coloured by entity."""
    panels = list(panel_titles)
    nrows = int(np.ceil(len(panels) / ncols))
    fig = make_subplots(
        rows=nrows, cols=min(ncols, len(panels)),
        subplot_titles=[panel_titles[p] for p in panels],
        horizontal_spacing=0.18, vertical_spacing=0.12,
    )
    for k, panel in enumerate(panels):
        row, col = k // ncols + 1, k % ncols + 1
        block = frame[frame[panel_col] == panel].sort_values(value)
        fig.add_trace(
            go.Scatter(
                x=block[value], y=block.arm, mode="markers", showlegend=False,
                marker=dict(color=_colors(block.entity), size=8),
                error_x=dict(
                    type="data", symmetric=False,
                    array=block[hi] - block[value], arrayminus=block[value] - block[lo],
                    thickness=1.2, width=0, color=GUIDE,
                ),
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ),
            row=row, col=col,
        )
        for x, label in reference.get(panel, []):
            if np.isfinite(x):
                add_reference_line(fig, x=x, label=label, row=row, col=col)
        fig.update_yaxes(
            categoryorder="array", categoryarray=list(block.arm), title_text="Feature arm",
            automargin=True, row=row, col=col,
        )
        fig.update_xaxes(title_text=x_title, row=row, col=col)
    _legend_dummies(fig, set(frame.entity))
    apply_theme(fig, height=height)
    return fig


def arm_scores(summary_dir: Path, out_root: Path, translator: str = "mlp") -> dict:
    """The verdict figure: every arm's held-out score, per cell line."""
    runs = pd.read_parquet(summary_dir / "runs.parquet")
    d = _with_entity(
        runs[(runs.scope == "full") & (runs.split == "grouped")
             & (runs.translator == translator)
             & (runs.target_source == "run1_all_controls")]
    )
    d = d[~d.arm.str.contains(SHUFFLE_TAG, regex=False) & d[VERDICT].notna()]
    lines = sorted(d.cell_line.unique())
    ref = {}
    for line in lines:
        block = runs[(runs.cell_line == line) & (runs.scope == "full")
                     & (runs.split == "grouped") & (runs.translator == translator)]
        ref[line] = [(_null_ceiling(block), "shuffled-label null")]
    fig = _dot_grid(
        d, VERDICT, VERDICT + "_lo", VERDICT + "_hi", "cell_line",
        {x: _line_label(x) for x in lines},
        "Fold-centred correlation (disattenuated)", ref, 640,
    )
    payload = d[[
        "cell_line", "arm", "entity", "family", "feature_dim", "n_perturbations",
        VERDICT, VERDICT + "_lo", VERDICT + "_hi", PREREG, PREREG + "_lo", PREREG + "_hi",
    ]].to_dict("records")
    save_figure_bundle(
        fig, "arm_scores",
        data={
            "records": payload, "metric": VERDICT, "preregistered_metric": PREREG,
            "translator": translator,
            "null_ceiling": {line: ref[line][0][0] for line in lines},
        },
        alt=(
            "Sorted dot plot of the fold-centred held-out correlation for every arm, one "
            "panel per cell line, with bootstrap intervals and the largest shuffled-label "
            "draw marked."
        ),
        root=out_root,
    )
    return {"figure": "arm_scores", "n_records": len(payload), "fig": fig}


def metric_check(summary_dir: Path, out_root: Path) -> dict:
    """Why the verdict moved off the pre-registered cell.

    Each arm appears once: its pre-registered score against its fold-centred
    score. The shuffled-label draws should sit at zero on both axes and only do
    so on one of them.
    """
    runs = pd.read_parquet(summary_dir / "runs.parquet")
    d = _with_entity(_primary(runs))
    d = d[d[PREREG].notna() & d[VERDICT].notna()]
    lines = sorted(d.cell_line.unique())
    fig = make_subplots(
        rows=1, cols=len(lines), subplot_titles=[_line_label(x) for x in lines],
        horizontal_spacing=0.16,
    )
    for k, line in enumerate(lines, start=1):
        block = d[d.cell_line == line]
        fig.add_trace(
            go.Scatter(
                x=block[PREREG], y=block[VERDICT], mode="markers", text=block.arm,
                showlegend=False,
                marker=dict(color=_colors(block.entity), size=8, opacity=0.85),
                hovertemplate="%{text}<br>pre-registered %{x:.4f}, "
                              "fold-centred %{y:.4f}<extra></extra>",
            ),
            row=1, col=k,
        )
        add_reference_line(fig, y=0.0, label="null at zero", row=1, col=k)
        fig.update_xaxes(title_text="Pre-registered residual correlation", row=1, col=k)
        fig.update_yaxes(title_text="Fold-centred correlation", row=1, col=k)
    _legend_dummies(fig, set(d.entity))
    apply_theme(fig, height=470)
    save_figure_bundle(
        fig, "metric_check",
        data={"records": d[["cell_line", "arm", "entity", PREREG, VERDICT]].to_dict("records")},
        alt=(
            "Scatter of the pre-registered residual correlation against the fold-centred "
            "correlation for every arm, one panel per cell line; the shuffled-label draws "
            "are away from zero on the horizontal axis and at zero on the vertical one."
        ),
        root=out_root,
    )
    return {"figure": "metric_check", "n_records": int(d.shape[0]), "fig": fig}


def vs_coexpression(summary_dir: Path, out_root: Path, translator: str = "mlp") -> dict:
    """Paired difference against co-expression on the same held-out perturbations."""
    con = pd.read_parquet(summary_dir / "contrasts.parquet")
    d = con[(con.scope == "full") & (con.split == "grouped")
            & (con.translator == translator) & (con.metric == "centered_weighted")
            & (con.target_source == "run1_all_controls")]
    d = _with_entity(d)
    d = d[~d.arm.str.contains(SHUFFLE_TAG, regex=False) & d.median_difference.notna()]
    lines = sorted(d.cell_line.unique())
    ref, resolvable = {}, {}
    for line in lines:
        mrd = float(np.nanmedian(
            d[d.cell_line == line].min_resolvable_difference_achieved.to_numpy()))
        ref[line] = [(0.0, "co-expression")]
        resolvable[line] = mrd
    fig = _dot_grid(
        d, "median_difference", "lo", "hi", "cell_line",
        {x: _line_label(x) for x in lines},
        "Paired difference against co-expression", ref, 600,
    )
    save_figure_bundle(
        fig, "vs_coexpression",
        data={"records": d.to_dict("records"), "translator": translator,
              "metric": "centered_weighted", "min_resolvable_difference": resolvable},
        alt=(
            "Paired difference between each arm and the co-expression arm on the same "
            "held-out perturbations, with bootstrap intervals; zero marks the "
            "co-expression baseline itself."
        ),
        root=out_root,
    )
    return {"figure": "vs_coexpression", "n_records": int(d.shape[0]), "fig": fig}


def split_gap(summary_dir: Path, out_root: Path) -> dict:
    """Grouped against random split: how much of a score is family memorisation."""
    runs = pd.read_parquet(summary_dir / "runs.parquet")
    d = runs[(runs.scope == "full") & (runs.translator == "mlp")
             & (runs.target_source == "run1_all_controls")]
    d = _with_entity(d)
    piv = d.pivot_table(index=["cell_line", "arm", "entity"], columns="split", values=VERDICT)
    piv = piv.dropna().reset_index()
    fig = go.Figure()
    lim = [float(min(piv["grouped"].min(), piv["random"].min())) - 0.02,
           float(max(piv["grouped"].max(), piv["random"].max())) + 0.02]
    fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                  line=dict(dash="dot", width=1, color=GUIDE))
    for ent in ENTITY_ORDER:
        sub = piv[piv.entity == ent]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["grouped"], y=sub["random"], mode="markers", name=ENTITY_LABEL[ent],
                marker=dict(color=COLOR[ent], size=8, opacity=0.85),
                text=sub.cell_line.map(_line_label) + ", " + sub.arm,
                hovertemplate="%{text}<br>grouped %{x:.4f}, random %{y:.4f}<extra></extra>",
            )
        )
    fig.update_xaxes(title_text="Grouped split (correlated perturbations held out together)")
    fig.update_yaxes(title_text="Random split")
    apply_theme(fig, height=480)
    save_figure_bundle(
        fig, "split_gap",
        data={"records": piv.to_dict("records"), "metric": VERDICT},
        alt=(
            "Each arm's fold-centred score under the grouped split against the random "
            "split; the dotted diagonal is equality, and points above it score better "
            "when related perturbed genes can appear on both sides."
        ),
        root=out_root,
    )
    return {"figure": "split_gap", "n_records": int(piv.shape[0]), "fig": fig}


def transfer_scores(summary_dir: Path, out_root: Path) -> dict | None:
    """Train on one screen, score on the other, against the gene-identity null."""
    path = summary_dir / "transfer_runs.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    d = _with_entity(d[(d.translator == "mlp") & (d.split == "cross_line__dest_reference")])
    d = d[d.scope.isin(["transfer_full", "transfer_unseen"]) & d[VERDICT].notna()]
    if d.empty:
        return None
    scope_label = {"transfer_full": "every shared gene", "transfer_unseen": "unseen genes only"}
    d = d.assign(panel=d.scope.map(scope_label) + " — " + d.cell_line.map(_direction_label))
    panels = [f"{scope_label[s]} — {_direction_label(c)}"
              for s in ("transfer_full", "transfer_unseen")
              for c in sorted(d.cell_line.unique())]
    panels = [p for p in panels if p in set(d.panel)]
    ref = {}
    for p in panels:
        gi = d[(d.panel == p) & (d.arm == "null_randgene_d5120")]
        ref[p] = [(float(gi[VERDICT].iloc[0]), "gene-identity null")] if not gi.empty else []
    fig = _dot_grid(
        d, VERDICT, VERDICT + "_lo", VERDICT + "_hi", "panel",
        {p: p for p in panels},
        "Fold-centred correlation on the held-out line", ref, 900, ncols=2,
    )
    save_figure_bundle(
        fig, "transfer",
        data={"records": d.to_dict("records"), "metric": VERDICT, "mode": "dest_reference"},
        alt=(
            "Cross-cell-line transfer in both directions, over every shared gene and over "
            "only the genes never perturbed in the training line, with the "
            "fixed-random-vector-per-gene null marked in each panel."
        ),
        root=out_root,
    )
    return {"figure": "transfer", "n_records": int(d.shape[0]), "fig": fig}


def transfer_mode_agreement(summary_dir: Path, out_root: Path) -> dict | None:
    """Does the choice of what the destination response is measured against matter."""
    path = summary_dir / "transfer_runs.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    d = _with_entity(d[d.translator == "mlp"])
    keys = ["cell_line", "scope", "arm", "entity"]
    piv = d.pivot_table(index=keys, columns="transfer_mode", values=VERDICT).reset_index()
    others = [m for m in ("source_only", "dest_all") if m in piv.columns]
    if "dest_reference" not in piv.columns or not others:
        return None
    label = {"source_only": "source-line mean", "dest_all": "all destination cells"}
    fig = make_subplots(
        rows=1, cols=len(others), subplot_titles=[label[m] for m in others],
        horizontal_spacing=0.16,
    )
    for k, mode in enumerate(others, start=1):
        block = piv.dropna(subset=["dest_reference", mode])
        fig.add_trace(
            go.Scatter(
                x=block["dest_reference"], y=block[mode], mode="markers", showlegend=False,
                marker=dict(color=_colors(block.entity), size=7, opacity=0.85),
                text=block.arm + ", " + block.scope.str.replace("transfer_", "", regex=False),
                hovertemplate="%{text}<br>held-out reference %{x:.4f}, "
                              "this mode %{y:.4f}<extra></extra>",
            ),
            row=1, col=k,
        )
        lim = [float(min(block["dest_reference"].min(), block[mode].min())) - 0.02,
               float(max(block["dest_reference"].max(), block[mode].max())) + 0.02]
        fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                      line=dict(dash="dot", width=1, color=GUIDE), row=1, col=k)
        fig.update_xaxes(title_text="Held-out reference cells", row=1, col=k)
        fig.update_yaxes(title_text=f"Reference: {label[mode]}", row=1, col=k)
    _legend_dummies(fig, set(piv.entity))
    apply_theme(fig, height=460)
    save_figure_bundle(
        fig, "transfer_mode_agreement",
        data={"records": piv.to_dict("records"), "metric": VERDICT},
        alt=(
            "Each transfer arm's score under the held-out-reference mode against its score "
            "under the two other choices of reference, with the equality line drawn."
        ),
        root=out_root,
    )
    return {"figure": "transfer_mode_agreement", "n_records": int(piv.shape[0]), "fig": fig}


def control_cell_overlap(summary_dir: Path, out_root: Path) -> dict | None:
    """Does sharing control cells with the target inflate the statistical baseline.

    ``ctrl1`` shares every control cell with the co-expression features and
    ``ctrl0`` shares none, so the difference is the size of that advantage. An
    arm that never touches a control cell should not move.
    """
    path = summary_dir / "control_cell_overlap.parquet"
    if not path.exists():
        return None
    d = _with_entity(pd.read_parquet(path))
    d = d[d.translator == "mlp"]
    piv = d.pivot_table(index=["cell_line", "arm", "entity"], columns="variant",
                        values=VERDICT).reset_index()
    if not {"ctrl0", "ctrl1"}.issubset(piv.columns):
        return None
    piv = piv.assign(shared_minus_disjoint=piv["ctrl1"] - piv["ctrl0"])
    piv = piv.dropna(subset=["shared_minus_disjoint"])
    lines = sorted(piv.cell_line.unique())
    fig = make_subplots(
        rows=1, cols=len(lines), subplot_titles=[_line_label(x) for x in lines],
        horizontal_spacing=0.18,
    )
    for k, line in enumerate(lines, start=1):
        block = piv[piv.cell_line == line].sort_values("shared_minus_disjoint")
        fig.add_trace(
            go.Scatter(
                x=block.shared_minus_disjoint, y=block.arm, mode="markers", showlegend=False,
                marker=dict(color=_colors(block.entity), size=9),
                hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
            ),
            row=1, col=k,
        )
        add_reference_line(fig, x=0.0, label="no advantage", row=1, col=k)
        fig.update_yaxes(categoryorder="array", categoryarray=list(block.arm),
                         title_text="Feature arm", automargin=True, row=1, col=k)
        fig.update_xaxes(title_text="Score with shared control cells minus score without",
                         row=1, col=k)
    _legend_dummies(fig, set(piv.entity))
    apply_theme(fig, height=440)
    save_figure_bundle(
        fig, "control_cell_overlap",
        data={"records": piv.to_dict("records"), "absolute": d.to_dict("records"),
              "metric": VERDICT},
        alt=(
            "Difference between each arm's score against a target built from the same "
            "control cells the co-expression features use and one built from the other "
            "half of the control cells, one panel per cell line."
        ),
        root=out_root,
    )
    return {"figure": "control_cell_overlap", "n_records": int(piv.shape[0]), "fig": fig}


def grid_cells(summary_dir: Path, out_root: Path, arms: list[str] | None = None) -> dict:
    """The whole correlation grid for the arms that carry the claim."""
    grid = pd.read_parquet(summary_dir / "grid.parquet")
    d = grid[(grid.scope == "full") & (grid.split == "grouped") & (grid.translator == "mlp")
             & (grid.target_source == "run1_all_controls")]
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
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
        fig.add_trace(
            go.Heatmap(
                z=pivot.to_numpy(), x=list(pivot.columns), y=list(pivot.index),
                coloraxis="coloraxis",
                hovertemplate="%{y}<br>%{x}: %{z:.4f}<extra></extra>",
            ),
            row=1, col=k,
        )
        fig.update_xaxes(title_text="Target / weighting / ceiling correction",
                         row=1, col=k, tickangle=40)
    fig.update_yaxes(title_text="Feature arm", automargin=True)
    apply_theme(fig, height=600)
    save_figure_bundle(
        fig, "grid_cells",
        data={"records": d.to_dict("records")},
        alt=(
            "Heatmap of all twelve cells of the correlation grid for every arm: the "
            "residual, fold-centred and raw targets, each weighted and unweighted, each "
            "with and without the ceiling correction."
        ),
        root=out_root,
    )
    return {"figure": "grid_cells", "n_records": int(d.shape[0]), "fig": fig}


def build_all(summary_dir: Path, out_root: Path | None = None) -> list[dict]:
    out_root = out_root or (paths.repo_dir() / "figures")
    made = [
        metric_check(summary_dir, out_root),
        arm_scores(summary_dir, out_root),
        vs_coexpression(summary_dir, out_root),
        split_gap(summary_dir, out_root),
        grid_cells(summary_dir, out_root),
        control_cell_overlap(summary_dir, out_root),
        transfer_scores(summary_dir, out_root),
        transfer_mode_agreement(summary_dir, out_root),
    ]
    return [m for m in made if m]
