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
from silico_figures import SEQUENTIAL, add_reference_line, apply_theme, save_figure_bundle

from .. import paths

#: guide lines (equality diagonals, error bars) are drawn in a light neutral that
#: is not one of the entity colours, so nothing in a legend shares its colour.
GUIDE = "#A8A399"

#: PREREG is the pre-registered decision rule and the cell the verdict is read
#: from. DISCRIM is the post-hoc fold-centred version of the same correlation,
#: reported beside it: it removes the held-out fold's own per-gene mean from
#: both sides, so it uses held-out outcomes and answers the narrower question of
#: how well an arm separates one held-out perturbation from another.
PREREG = "residual__weighted__disattenuated"
DISCRIM = "centered__weighted__disattenuated"

SHUFFLE_TAG = "+shuffled_labels_d"

ENTITY_LABEL = {
    "scgpt": "scGPT",
    "scprint": "scPRINT",
    "transcriptformer": "TranscriptFormer",
    "aido_cell": "AIDO.Cell",
    "esm2": "ESM2, protein only",
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
    if arm.startswith("esm2"):
        return "esm2"
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


def _null_ceiling(runs: pd.DataFrame, col: str = DISCRIM) -> float:
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


#: how each metric is named on a figure
METRIC_LABEL = {PREREG: "pre-registered", DISCRIM: "fold-centred (post hoc)"}


def _both_metrics(d: pd.DataFrame, lines: list[str]) -> tuple:
    """One long frame carrying both metrics, and the panel order for it.

    The verdict is read from the pre-registered cell; the fold-centred cell is
    reported beside it rather than instead of it, so both are drawn.
    """
    pieces = []
    for col in (PREREG, DISCRIM):
        piece = d.copy()
        piece["value"] = piece[col]
        piece["lo"] = piece[col + "_lo"]
        piece["hi"] = piece[col + "_hi"]
        piece["panel"] = piece.cell_line + "|" + col
        pieces.append(piece)
    long = pd.concat(pieces, ignore_index=True)
    panels = [f"{line}|{col}" for col in (PREREG, DISCRIM) for line in lines]
    titles = {
        f"{line}|{col}": f"{_line_label(line)}, {METRIC_LABEL[col]}"
        for col in (PREREG, DISCRIM)
        for line in lines
    }
    return long, panels, titles


def arm_scores(summary_dir: Path, out_root: Path, translator: str = "mlp") -> dict:
    """The verdict figure: every arm's held-out score, per cell line."""
    runs = pd.read_parquet(summary_dir / "runs.parquet")
    d = _with_entity(
        runs[(runs.scope == "full") & (runs.split == "grouped")
             & (runs.translator == translator)
             & (runs.target_source == "run1_all_controls")]
    )
    d = d[~d.arm.str.contains(SHUFFLE_TAG, regex=False) & d[DISCRIM].notna()]
    lines = sorted(d.cell_line.unique())
    long, panels, titles = _both_metrics(d, lines)
    ref = {}
    for line in lines:
        block = runs[(runs.cell_line == line) & (runs.scope == "full")
                     & (runs.split == "grouped") & (runs.translator == translator)]
        for col in (PREREG, DISCRIM):
            ref[f"{line}|{col}"] = [(_null_ceiling(block, col), "shuffled-label null")]
    fig = _dot_grid(
        long, "value", "lo", "hi", "panel", {k: titles[k] for k in panels},
        "Held-out correlation (disattenuated)", ref, 1180,
    )
    payload = d[[
        "cell_line", "arm", "entity", "family", "feature_dim", "n_perturbations",
        DISCRIM, DISCRIM + "_lo", DISCRIM + "_hi", PREREG, PREREG + "_lo", PREREG + "_hi",
    ]].to_dict("records")
    save_figure_bundle(
        fig, "arm_scores",
        data={
            "records": payload, "metric": PREREG, "discrimination_metric": DISCRIM,
            "translator": translator,
            "null_ceiling": {
                f"{line}|{col}": ref[f"{line}|{col}"][0][0]
                for line in lines for col in (PREREG, DISCRIM)
            },
        },
        alt=(
            "Sorted dot plot of every arm's held-out correlation with group-resampled "
            "intervals, four panels: each cell line under the pre-registered rule and "
            "under the post-hoc fold-centred version, with the largest shuffled-label "
            "draw marked in each."
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
    d = d[d[PREREG].notna() & d[DISCRIM].notna()]
    lines = sorted(d.cell_line.unique())
    fig = make_subplots(
        rows=1, cols=len(lines), subplot_titles=[_line_label(x) for x in lines],
        horizontal_spacing=0.16,
    )
    for k, line in enumerate(lines, start=1):
        block = d[d.cell_line == line]
        fig.add_trace(
            go.Scatter(
                x=block[PREREG], y=block[DISCRIM], mode="markers", text=block.arm,
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
        data={"records": d[["cell_line", "arm", "entity", PREREG, DISCRIM]].to_dict("records")},
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
            & (con.translator == translator) & (con.metric == "residual_weighted")
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
              "metric": "residual_weighted", "min_resolvable_difference": resolvable},
        alt=(
            "Paired difference between each arm and the co-expression arm on the same "
            "held-out perturbations, with bootstrap intervals; zero marks the "
            "co-expression baseline itself."
        ),
        root=out_root,
    )
    return {"figure": "vs_coexpression", "n_records": int(d.shape[0]), "fig": fig}


def discrimination(summary_dir: Path, out_root: Path, translator: str = "mlp") -> dict:
    """The second scoring axis: is a prediction closer to its own perturbation?

    No residualization anywhere. The prediction is put back on the effect scale
    by adding the training fold's mean effect, and Arc Institute's ``cell-eval``
    ranks its distance to every held-out perturbation's measured profile.
    Chance is 0.5.
    """
    runs = pd.read_parquet(summary_dir / "runs.parquet")
    d = _with_entity(_primary(runs)[lambda x: x.translator == translator])
    d = d[~d.arm.str.contains(SHUFFLE_TAG, regex=False)]
    d = d[d.discrimination_cell_eval_mean.notna()]
    if d.empty:
        return {}
    lines = sorted(d.cell_line.unique())
    ref = {line: [(0.5, "chance")] for line in lines}
    fig = _dot_grid(
        d.assign(_lo=d.discrimination_cell_eval_mean, _hi=d.discrimination_cell_eval_mean),
        "discrimination_cell_eval_mean", "_lo", "_hi", "cell_line",
        {x: _line_label(x) for x in lines},
        "Perturbation discrimination (cell-eval, effect scale)", ref, 640,
    )
    payload = d[[
        "cell_line", "arm", "entity", "family", "cell_eval_version",
        "discrimination_cell_eval_mean", "discrimination_cell_eval_median",
        "discrimination_ours_residual_mean", DISCRIM, PREREG,
    ]].to_dict("records")
    save_figure_bundle(
        fig, "discrimination",
        data={"records": payload, "chance": 0.5, "translator": translator,
              "package": "cell-eval",
              "versions": sorted(set(d.cell_eval_version.dropna().astype(str)))},
        alt=(
            "Sorted dot plot of perturbation discrimination for every arm, one panel per "
            "cell line, with the 0.5 chance line marked; embedding arms sit above chance "
            "and co-expression and response-only arms at or below it."
        ),
        root=out_root,
    )
    return {"figure": "discrimination", "n_records": len(payload), "fig": fig}


def anchor_sensitivity(summary_dir: Path, out_root: Path) -> dict | None:
    """Does the query the model was asked change what the response features carry?"""
    path = summary_dir / "sensitivity.parquet"
    if not path.exists():
        return None
    s = pd.read_parquet(path)
    s = s[(s.translator == "mlp") & s.arm.str.contains("_resp")]
    if s.empty:
        return None
    s = s.assign(
        model=[a.split("@")[0].replace("_resp", "") for a in s.arm],
        anchor=[a.split("@")[1] if "@" in a else "mean_control" for a in s.arm],
    )
    anchors = ["mean_control", "uniform_virtual", "individual_cells", "mean_control_masked"]
    anchors = [a for a in anchors if a in set(s.anchor)]
    models = [m for m in ENTITY_ORDER if m in set(s.model)]
    fig = go.Figure()
    for m in models:
        sub = s[s.model == m].set_index("anchor").reindex(anchors).reset_index()
        fig.add_trace(
            go.Scatter(
                x=sub[DISCRIM], y=sub["anchor"], mode="markers", name=ENTITY_LABEL[m],
                marker=dict(color=COLOR[m], size=9),
                error_x=dict(
                    type="data", symmetric=False,
                    array=sub[DISCRIM + "_hi"] - sub[DISCRIM],
                    arrayminus=sub[DISCRIM] - sub[DISCRIM + "_lo"],
                    thickness=1.2, width=0, color=GUIDE,
                ),
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            )
        )
    runs = pd.read_parquet(summary_dir / "runs.parquet")
    null = _null_ceiling(_primary(runs)[lambda x: x.cell_line == "hepg2"])
    add_reference_line(fig, x=null, label="shuffled-label null")
    fig.update_xaxes(title_text="Fold-centred correlation (disattenuated)")
    fig.update_yaxes(title_text="Anchor the model was queried at", automargin=True)
    apply_theme(fig, height=420)
    save_figure_bundle(
        fig, "anchor_sensitivity",
        data={"records": s.to_dict("records"), "metric": DISCRIM,
              "cell_line": "hepg2", "null_ceiling": null},
        alt=(
            "Response-only score in HepG2 for each model at four query anchors: the "
            "on-distribution mean control, the uniform virtual cell, individual control "
            "cells, and the masked read-out, against the shuffled-label null."
        ),
        root=out_root,
    )
    return {"figure": "anchor_sensitivity", "n_records": int(s.shape[0]), "fig": fig}


def split_gap(summary_dir: Path, out_root: Path) -> dict:
    """Grouped against random split: how much of a score is family memorisation."""
    runs = pd.read_parquet(summary_dir / "runs.parquet")
    d = runs[(runs.scope == "full") & (runs.translator == "mlp")
             & (runs.target_source == "run1_all_controls")]
    d = _with_entity(d)
    piv = d.pivot_table(index=["cell_line", "arm", "entity"], columns="split", values=DISCRIM)
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
        data={"records": piv.to_dict("records"), "metric": DISCRIM},
        alt=(
            "Each arm's fold-centred score under the grouped split against the random "
            "split; the dotted diagonal is equality, and points above it score better "
            "when related perturbed genes can appear on both sides."
        ),
        root=out_root,
    )
    return {"figure": "split_gap", "n_records": int(piv.shape[0]), "fig": fig}


#: the arms the protein control is read against: the embedding arms whose tables
#: may come from a protein model, the statistical baseline, and the two nulls
#: matched to the protein arms' widths
PROTEIN_PANEL_ARMS = (
    "esm2_3b_emb", "esm2_650m_emb", "transcriptformer_emb", "scprint_emb", "scgpt_emb",
    "aido_cell_emb", "coexp", "null_randgene_d5120", "null_randgene_d2560",
)


def protein_arms(summary_dir: Path, out_root: Path, translator: str = "mlp") -> dict | None:
    """The raw protein arm beside the arms whose tables came from a protein model."""
    runs = pd.read_parquet(summary_dir / "runs.parquet")
    if not runs.arm.str.startswith("esm2").any():
        return None
    d = _with_entity(
        runs[(runs.scope == "full") & (runs.split == "grouped")
             & (runs.translator == translator)
             & (runs.target_source == "run1_all_controls")
             & runs.arm.isin(PROTEIN_PANEL_ARMS) & runs[DISCRIM].notna()]
    )
    if d.empty:
        return None
    lines = sorted(d.cell_line.unique())
    long, panels, titles = _both_metrics(d, lines)
    ref = {}
    for line in lines:
        block = runs[(runs.cell_line == line) & (runs.scope == "full")
                     & (runs.split == "grouped") & (runs.translator == translator)]
        for col in (PREREG, DISCRIM):
            ref[f"{line}|{col}"] = [(_null_ceiling(block, col), "shuffled-label null")]
    fig = _dot_grid(
        long, "value", "lo", "hi", "panel", {k: titles[k] for k in panels},
        "Held-out correlation (disattenuated)", ref, 900,
    )
    payload = d[[
        "cell_line", "arm", "entity", "family", "feature_dim", "n_perturbations",
        DISCRIM, DISCRIM + "_lo", DISCRIM + "_hi", PREREG, PREREG + "_lo", PREREG + "_hi",
    ]].to_dict("records")
    save_figure_bundle(
        fig, "protein_arms",
        data={"records": payload, "metric": PREREG, "discrimination_metric": DISCRIM,
              "translator": translator,
              "null_ceiling": {
                  f"{line}|{col}": ref[f"{line}|{col}"][0][0]
                  for line in lines for col in (PREREG, DISCRIM)
              }},
        alt=(
            "Sorted dot plot of the held-out correlation for the raw ESM2 arms, the four "
            "model embedding arms, co-expression and the two width-matched random controls, "
            "four panels: each cell line under the pre-registered rule and under the "
            "post-hoc fold-centred version."
        ),
        root=out_root,
    )
    return {"figure": "protein_arms", "n_records": len(payload), "fig": fig}


#: how the similarity matrix orders and names its tables
TABLE_LABEL = {
    "esm2_3b": "ESM2-3B",
    "esm2_650m": "ESM2-650M",
    "transcriptformer": "TranscriptFormer",
    "scprint": "scPRINT",
    "scgpt": "scGPT",
    "aido_cell": "AIDO.Cell",
}


def table_similarity(summary_dir: Path, out_root: Path) -> dict | None:
    """How much of the raw protein table survives in each model's gene table.

    Read with representational similarity: the Spearman correlation between two
    tables' gene-gene cosine similarities. Linear CKA is kept in the data file
    but not plotted, because its floor moves with width -- two independent
    random tables at 1,280 and 2,560 dimensions score 0.3934 on it, which is
    the same as a real pair -- while the rank-similarity floor is at zero for
    every width.
    """
    path = summary_dir / "table_similarity.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    lines = [x for x in ("hepg2", "jurkat") if x in record.get("lines", {})]
    if not lines:
        return None
    names = [k for k in TABLE_LABEL if any(
        k in {p.split("|")[0], p.split("|")[1]}
        for p in record["lines"][lines[0]]["pairs"]
    )]
    fig = make_subplots(
        rows=1, cols=len(lines), subplot_titles=[_line_label(x) for x in lines],
        horizontal_spacing=0.16,
    )
    payload = []
    for k, line in enumerate(lines):
        pairs = record["lines"][line]["pairs"]
        widths = record["lines"][line]["widths"]

        def value(a: str, b: str, stat: str, pairs=pairs) -> float:
            if a == b:
                return 1.0
            key = f"{a}|{b}" if f"{a}|{b}" in pairs else f"{b}|{a}"
            return float(pairs[key][stat]) if key in pairs else float("nan")

        z = [[value(a, b, "rsa_spearman") for b in names] for a in names]
        cka = [[value(a, b, "cka") for b in names] for a in names]
        rand = [p for p in pairs if "random_d" in p]
        floor = max(abs(float(pairs[p]["rsa_spearman"])) for p in rand) if rand else float("nan")
        cka_floor = max(
            float(pairs[p]["cka"]) for p in rand
            if p.split("|")[0].startswith("random") and p.split("|")[1].startswith("random")
        ) if rand else float("nan")
        payload.append({"cell_line": line, "names": names, "rsa_spearman": z, "cka": cka,
                        "widths": widths, "random_table_floor_rsa": floor,
                        "random_pair_ceiling_cka": cka_floor,
                        "n_genes_compared": record["lines"][line]["n_genes_compared"],
                        "n_genes_rsa": record["lines"][line]["n_genes_rsa"]})
        fig.add_trace(
            go.Heatmap(
                z=z, x=[TABLE_LABEL[n] for n in names], y=[TABLE_LABEL[n] for n in names],
                zmin=0, zmax=0.9, colorscale=SEQUENTIAL, showscale=(k == len(lines) - 1),
                colorbar=dict(title="Rank<br>similarity"),
                text=[[f"{v:.2f}" for v in row] for row in z],
                texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}: %{z:.4f}<extra></extra>",
            ),
            row=1, col=k + 1,
        )
        fig.update_xaxes(title_text="Gene table", row=1, col=k + 1)
        fig.update_yaxes(title_text="Gene table", autorange="reversed", row=1, col=k + 1)
    apply_theme(fig, height=460)
    save_figure_bundle(
        fig, "table_similarity",
        data={"panels": payload,
              "metric": ("Spearman correlation between the two tables' gene-gene cosine "
                         "similarities, over a fixed 1,500-gene subset"),
              "secondary_metric": "linear CKA over every read-out gene, not plotted"},
        alt=(
            "Two heatmaps, one per cell line, of the rank similarity between every pair of "
            "gene tables: the two raw ESM2 tables and the four single-cell models'. The two "
            "ESM2 sizes agree strongly, TranscriptFormer and scPRINT keep a clear protein "
            "signature, and scGPT and AIDO.Cell sit at the random-table floor."
        ),
        root=out_root,
    )
    return {"figure": "table_similarity", "n_records": len(payload), "fig": fig}


def transfer_scores(summary_dir: Path, out_root: Path) -> dict | None:
    """Train on one screen, score on the other, against the gene-identity null."""
    path = summary_dir / "transfer_runs.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    d = _with_entity(d[(d.translator == "mlp") & (d.split == "cross_line__dest_reference")])
    d = d[d.scope.isin(["transfer_full", "transfer_unseen"]) & d[DISCRIM].notna()]
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
        ref[p] = [(float(gi[PREREG].iloc[0]), "gene-identity null")] if not gi.empty else []
    fig = _dot_grid(
        d, PREREG, PREREG + "_lo", PREREG + "_hi", "panel",
        {p: p for p in panels},
        "Pre-registered correlation on the held-out line", ref, 900, ncols=2,
    )
    save_figure_bundle(
        fig, "transfer",
        data={"records": d.to_dict("records"), "metric": PREREG,
              "discrimination_metric": DISCRIM, "mode": "dest_reference"},
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
    piv = d.pivot_table(index=keys, columns="transfer_mode", values=DISCRIM).reset_index()
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
        data={"records": piv.to_dict("records"), "metric": DISCRIM},
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
                        values=DISCRIM).reset_index()
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
              "metric": DISCRIM},
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
        discrimination(summary_dir, out_root),
        anchor_sensitivity(summary_dir, out_root),
        concat_agreement(summary_dir, out_root),
        split_gap(summary_dir, out_root),
        protein_arms(summary_dir, out_root),
        table_similarity(summary_dir, out_root),
        grid_cells(summary_dir, out_root),
        control_cell_overlap(summary_dir, out_root),
        transfer_scores(summary_dir, out_root),
        transfer_mode_agreement(summary_dir, out_root),
    ]
    return [m for m in made if m]


def concat_agreement(summary_dir: Path, out_root: Path) -> dict | None:
    """Whether the two gene tables are right about the same held-out perturbations.

    One point per held-out perturbation: the score the scGPT-table arm reached
    on it against the score the raw-protein-table arm reached on the same one,
    under the same folds. Two arms can reach the same average either by being
    right about the same perturbations or about different ones, and only this
    view separates those.
    """
    pp_path = summary_dir / "per_perturbation.parquet"
    agree_path = summary_dir / "arm_agreement.parquet"
    if not pp_path.exists() or not agree_path.exists():
        return None
    a, b = "scgpt_emb", "esm2_3b_emb"
    pp = pd.read_parquet(pp_path)
    d = pp[
        (pp.scope == "full") & (pp.split == "grouped") & (pp.translator == "mlp")
        & (pp.target_source == "run1_all_controls") & pp.arm.isin([a, b])
    ]
    if d.empty:
        return None
    agree = pd.read_parquet(agree_path)
    lines = [ln for ln in ("hepg2", "jurkat") if (d.cell_line == ln).any()]
    fig = make_subplots(
        rows=1, cols=len(lines), subplot_titles=[_line_label(ln) for ln in lines],
        horizontal_spacing=0.09,
    )
    records = []
    for i, line in enumerate(lines, start=1):
        piv = (
            d[d.cell_line == line]
            .pivot_table(index="perturbation", columns="arm", values="residual_weighted")
            .dropna()
        )
        if piv.empty or a not in piv or b not in piv:
            continue
        lo = float(min(piv[a].min(), piv[b].min())) - 0.02
        hi = float(max(piv[a].max(), piv[b].max())) + 0.02
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, row=1, col=i,
                      line=dict(dash="dot", width=1, color=GUIDE))
        fig.add_trace(
            go.Scatter(
                x=piv[a], y=piv[b], mode="markers", showlegend=False,
                marker=dict(color=COLOR["scgpt"], size=5, opacity=0.5,
                            line=dict(width=0)),
                text=piv.index,
                hovertemplate="%{text}<br>scGPT %{x:.3f}, ESM2-3B %{y:.3f}<extra></extra>",
            ),
            row=1, col=i,
        )
        row = agree[
            (agree.cell_line == line) & (agree.split == "grouped")
            & (agree.translator == "mlp") & (agree.arm_a == a) & (agree.arm_b == b)
            & (agree.metric == "residual_weighted")
        ]
        if not row.empty:
            r = row.iloc[0]
            fig.add_annotation(
                x=0.02, y=0.98,
                xref=f"x{'' if i == 1 else i} domain",
                yref=f"y{'' if i == 1 else i} domain",
                xanchor="left", yanchor="top", showarrow=False,
                text=f"Spearman {r.spearman:.3f}, n = {int(r.n_perturbations)}",
            )
            records.append({"cell_line": line, "spearman": float(r.spearman),
                            "pearson": float(r.pearson),
                            "n_perturbations": int(r.n_perturbations)})
        fig.update_xaxes(title_text="scGPT table", row=1, col=i)
        fig.update_yaxes(title_text="ESM2-3B table" if i == 1 else None, row=1, col=i)
    apply_theme(fig, height=430)
    save_figure_bundle(
        fig, "concat_agreement",
        data={"records": records, "metric": "residual_weighted",
              "cell": "full scope, grouped split, MLP translator"},
        alt=(
            "One point per held-out perturbation, the scGPT-table arm's score against the "
            "raw-protein-table arm's score on the same perturbation, per cell line, with "
            "the rank correlation printed in each panel and a dotted equality diagonal."
        ),
        root=out_root,
    )
    return {"figure": "concat_agreement", "n_records": int(d.shape[0]), "fig": fig}
