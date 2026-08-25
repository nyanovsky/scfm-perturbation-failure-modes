"""Figures, built pod-side from the summary tables the training jobs produced.

Each figure reads ``results/summary/*.parquet`` and saves a bundle under
``figures/<name>/``, so it stays re-renderable from its own data without
rerunning anything.

The verdict is read on the pre-registered cell of run 3's grid (residual
target, reliability-weighted, ceiling-corrected). The fold-centred cell travels
with it in every bundle's data and is stated beside it, never in place of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle

from .. import config as C
from .. import paths

VERDICT = ("residual", "weighted")
DIAGNOSTIC = ("centered", "weighted")
#: same order of preference as ``analysis.DIRECT_BASELINES``
DIRECT_BASELINES = ("measured_control_mean", "unflagged_model_output")

#: guide marks are drawn in a neutral that is not one of the entity colours
GUIDE = "#A8A399"

ARM_LABEL = {
    "own_trainable": "scGPT table, trainable",
    "own_frozen": "scGPT table, frozen",
    "esm2_trainable": "ESM2 table, trainable",
    "esm2_frozen": "ESM2 table, frozen",
    "rand_frozen": "random table, frozen",
}
LINE_LABEL = {"hepg2": "HepG2", "jurkat": "Jurkat"}


def entity_colors() -> dict:
    raw = json.loads((paths.repo_dir() / "figures" / "entity_colors.json").read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


COLOR = entity_colors()


def tint(hex_colour: str, amount: float) -> str:
    """Blend a colour toward white.

    Two series that carry the same meaning-bearing hue (both are the same gene
    table) but a different reading of it are separated by lightness as well as
    by marker, so the pair stays distinguishable without colour vision.
    """
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    m = lambda c: int(round(c + (255 - c) * amount))  # noqa: E731
    return f"#{m(r):02X}{m(g):02X}{m(b):02X}"


def entity_of_arm(arm: str) -> str:
    """Which compared thing a training arm is built from."""
    if arm.startswith("own"):
        return "scgpt"
    if arm.startswith("esm2"):
        return "esm2"
    return "null"


def _verdict(restricted: pd.DataFrame, target=VERDICT) -> pd.DataFrame:
    return restricted[
        (restricted.target == target[0])
        & (restricted.weighting == target[1])
        & (restricted.translator == "mlp")
    ]


def _score(frame: pd.DataFrame, cell_line: str, arm: str) -> dict | None:
    sub = frame[(frame.cell_line == cell_line) & (frame.arm == arm) & (~frame.is_shuffled)]
    if sub.empty:
        return None
    r = sub.iloc[0]
    return {
        "score": float(r["disattenuated"]),
        "lo": float(r["disattenuated_lo"]),
        "hi": float(r["disattenuated_hi"]),
        "n_groups": int(r["n_groups"]),
    }


def null_band(band: pd.DataFrame, cell_line: str, base_arm: str) -> tuple[float, float] | None:
    sub = band[
        (band.cell_line == cell_line)
        & (band.base_arm == base_arm)
        & (band.translator == "mlp")
        & (band.target == VERDICT[0])
        & (band.weighting == VERDICT[1])
    ]
    if sub.empty:
        return None
    r = sub.iloc[0]
    return float(r["null_min"]), float(r["null_max"])


def pooled_null(band: pd.DataFrame, cell_line: str) -> tuple[float, float] | None:
    """The widest shuffled-label band measured in this cell line.

    Several arms carry their own draws; the band a score has to clear is the
    outer envelope of them, not the one that happens to be narrowest.
    """
    sub = band[
        (band.cell_line == cell_line)
        & (band.translator == "mlp")
        & (band.target == VERDICT[0])
        & (band.weighting == VERDICT[1])
    ]
    if sub.empty:
        return None
    return float(sub["null_min"].min()), float(sub["null_max"].max())


# --------------------------------------------------------------------------- #
# figure 1: the swap, before and after fine-tuning
# --------------------------------------------------------------------------- #
def figure_swap(restricted: pd.DataFrame, band: pd.DataFrame, out: Path) -> dict:
    """Response features read off each checkpoint, before and after training."""
    v = _verdict(restricted)
    rows, data = [], []
    for line in C.CELL_LINES:
        for arm in [a.name for a in C.ARMS]:
            before = _score(v, line, f"pre_{arm}_resp")
            after = _score(v, line, f"ft_{arm}_resp")
            if before is None and after is None:
                continue
            rows.append((line, arm, before, after))
            data.append(
                {
                    "cell_line": line,
                    "arm": arm,
                    "before": before,
                    "after": after,
                    "entity": entity_of_arm(arm),
                }
            )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[LINE_LABEL[c] for c in C.CELL_LINES],
        shared_xaxes=False,
        horizontal_spacing=0.16,
    )
    order = [a.name for a in C.ARMS]
    ypos = {a: len(order) - i for i, a in enumerate(order)}
    seen_legend = set()
    for col, line in enumerate(C.CELL_LINES, start=1):
        nb = pooled_null(band, line)
        if nb is not None:
            fig.add_vrect(
                x0=nb[0], x1=nb[1], row=1, col=col,
                fillcolor=COLOR["null"], opacity=0.12, line_width=0, layer="below",
            )
        for line_name, arm, before, after in rows:
            if line_name != line:
                continue
            colour = COLOR[entity_of_arm(arm)]
            y = ypos[arm]
            if before and after:
                fig.add_trace(
                    go.Scatter(
                        x=[before["score"], after["score"]], y=[y, y], mode="lines",
                        line=dict(color=colour, width=1.5), opacity=0.5,
                        hoverinfo="skip", showlegend=False,
                    ),
                    row=1, col=col,
                )
            for state, point, symbol, shade in (
                ("before", before, "circle-open", 0.55),
                ("after", after, "circle", 0.0),
            ):
                if point is None:
                    continue
                name = f"{state} fine-tuning"
                mcolour = tint(colour, shade)
                fig.add_trace(
                    go.Scatter(
                        x=[point["score"]], y=[y], mode="markers",
                        marker=dict(color=mcolour, size=10, symbol=symbol,
                                    line=dict(color=mcolour, width=1.5)),
                        error_x=dict(
                            type="data", symmetric=False,
                            array=[point["hi"] - point["score"]],
                            arrayminus=[point["score"] - point["lo"]],
                            color=GUIDE, thickness=1, width=0,
                        ),
                        name=name, legendgroup=name,
                        showlegend=name not in seen_legend,
                        hovertemplate=(
                            f"{ARM_LABEL[arm]}<br>{name}<br>"
                            "score %{x:.4f}<extra></extra>"
                        ),
                    ),
                    row=1, col=col,
                )
                seen_legend.add(name)
        for ref_arm, label in (("coexp", "co-expression"), ("scgpt_emb", "scGPT gene table")):
            r = _score(v, line, ref_arm)
            if r:
                add_reference_line(fig, x=r["score"], label=label, row=1, col=col)
        fig.update_yaxes(
            tickmode="array", tickvals=[ypos[a] for a in order],
            ticktext=[ARM_LABEL[a] for a in order],
            range=[0.4, len(order) + 0.6], title_text="" if col == 2 else "Training arm",
            row=1, col=col,
        )
        fig.update_xaxes(title_text="Response-feature score", row=1, col=col)
    apply_theme(fig, height=430)
    save_figure_bundle(
        fig, "swap_before_after", data={"rows": data, "metric": VERDICT}, root=paths.repo_dir() / "figures",
        alt=(
            "Two panels, HepG2 and Jurkat. For each of five training arms, the score of the "
            "response features read off the model before and after fine-tuning, against the "
            "shuffled-label band and the co-expression and gene-table reference lines."
        ),
    )
    return {"fig": fig, "rows": data}


# --------------------------------------------------------------------------- #
# figure 2: what the model predicts against what its features carry
# --------------------------------------------------------------------------- #
def figure_predict_vs_features(
    restricted: pd.DataFrame, direct: pd.DataFrame, band: pd.DataFrame, out: Path
) -> dict:
    """Three quantities per fine-tuned model, on one axis."""
    v = _verdict(restricted)
    d = direct[
        (direct.target == VERDICT[0]) & (direct.weighting == VERDICT[1])
    ] if not direct.empty else pd.DataFrame()

    series = [
        ("direct", "the model's predicted response", "diamond", 0.0),
        ("own_query", "response features, own query", "square", 0.35),
        ("value_substitution", "response features, value query", "circle-open", 0.65),
    ]
    data = []
    for line in C.CELL_LINES:
        for arm in [a.name for a in C.ARMS]:
            entry = {"cell_line": line, "arm": arm, "entity": entity_of_arm(arm)}
            entry["value_substitution"] = _score(v, line, f"ft_{arm}_resp")
            entry["own_query"] = _score(v, line, f"ftq_{arm}_resp")
            if not d.empty:
                sub = d[(d.cell_line == line) & (d.arm_label == f"direct_prediction/{arm}")]
                if "baseline" in sub.columns and not sub.empty:
                    for b in DIRECT_BASELINES:
                        pick = sub[sub.baseline == b]
                        if not pick.empty:
                            sub = pick
                            break
                if len(sub) > 1:
                    raise RuntimeError(
                        f"{line}/{arm}: {len(sub)} direct-prediction rows for one cell of the "
                        f"grid; baselines present = {sorted(set(sub.get('baseline', [])))}"
                    )
                entry["direct"] = (
                    {
                        "score": float(sub.iloc[0]["disattenuated"]),
                        "lo": float(sub.iloc[0]["disattenuated_lo"]),
                        "hi": float(sub.iloc[0]["disattenuated_hi"]),
                        "baseline": str(sub.iloc[0].get("baseline", "unflagged_model_output")),
                    }
                    if not sub.empty
                    else None
                )
            data.append(entry)

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=[LINE_LABEL[c] for c in C.CELL_LINES],
        horizontal_spacing=0.16,
    )
    order = [a.name for a in C.ARMS]
    ypos = {a: len(order) - i for i, a in enumerate(order)}
    seen = set()
    for col, line in enumerate(C.CELL_LINES, start=1):
        nb = pooled_null(band, line)
        if nb is not None:
            fig.add_vrect(
                x0=nb[0], x1=nb[1], row=1, col=col,
                fillcolor=COLOR["null"], opacity=0.12, line_width=0, layer="below",
            )
        for entry in data:
            if entry["cell_line"] != line:
                continue
            colour = COLOR[entry["entity"]]
            for key, label, symbol, shade in series:
                point = entry.get(key)
                if not point:
                    continue
                off = {"direct": 0.22, "own_query": 0.0, "value_substitution": -0.22}[key]
                mcolour = tint(colour, shade)
                fig.add_trace(
                    go.Scatter(
                        x=[point["score"]], y=[ypos[entry["arm"]] + off], mode="markers",
                        marker=dict(color=mcolour, size=9, symbol=symbol,
                                    line=dict(color=mcolour, width=1.5)),
                        name=label, legendgroup=label, showlegend=label not in seen,
                        hovertemplate=(
                            f"{ARM_LABEL[entry['arm']]}<br>{label}<br>"
                            "score %{x:.4f}<extra></extra>"
                        ),
                    ),
                    row=1, col=col,
                )
                seen.add(label)
        r = _score(v, line, "coexp")
        if r:
            add_reference_line(fig, x=r["score"], label="co-expression", row=1, col=col)
        fig.update_yaxes(
            tickmode="array", tickvals=[ypos[a] for a in order],
            ticktext=[ARM_LABEL[a] for a in order],
            range=[0.4, len(order) + 0.6],
            title_text="" if col == 2 else "Training arm", row=1, col=col,
        )
        fig.update_xaxes(title_text="Score against the measured response", row=1, col=col)
    apply_theme(fig, height=430)
    save_figure_bundle(
        fig, "predicted_and_features", data={"rows": data, "metric": VERDICT},
        root=paths.repo_dir() / "figures",
        alt=(
            "For each fine-tuned model, three scores on one axis: the model's own predicted "
            "perturbation response, and the response features read from it through its own "
            "query and through the value-substitution query, against the shuffled-label band."
        ),
    )
    return {"fig": fig, "rows": data}


# --------------------------------------------------------------------------- #
# figure 3: what fine-tuning did to the training objective
# --------------------------------------------------------------------------- #
def figure_training(epochs: pd.DataFrame, runs: pd.DataFrame, out: Path) -> dict:
    """Validation loss per epoch, and the agreement the recipe does not select on."""
    if epochs.empty:
        return {}
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"{LINE_LABEL[c]}, validation loss" for c in C.CELL_LINES
        ] + [f"{LINE_LABEL[c]}, response agreement" for c in C.CELL_LINES],
        vertical_spacing=0.16, horizontal_spacing=0.14,
    )
    seen = set()
    for col, line in enumerate(C.CELL_LINES, start=1):
        for arm in [a.name for a in C.ARMS]:
            sub = epochs[(epochs.cell_line == line) & (epochs.arm == arm)].sort_values("epoch")
            if sub.empty:
                continue
            colour = COLOR[entity_of_arm(arm)]
            dash = "solid" if arm.endswith("trainable") else "dot"
            label = ARM_LABEL[arm]
            fig.add_trace(
                go.Scatter(
                    x=sub["epoch"], y=sub["val_loss"], mode="lines+markers",
                    line=dict(color=colour, width=2, dash=dash), marker=dict(size=5),
                    name=label, legendgroup=label, showlegend=label not in seen,
                    hovertemplate="epoch %{x}<br>validation loss %{y:.4f}<extra></extra>",
                ),
                row=1, col=col,
            )
            seen.add(label)
            if "delta_ctrlmean_centered" in sub:
                fig.add_trace(
                    go.Scatter(
                        x=sub["epoch"], y=sub["delta_ctrlmean_centered"], mode="lines+markers",
                        line=dict(color=colour, width=2, dash=dash), marker=dict(size=5),
                        name=label, legendgroup=label, showlegend=False,
                        hovertemplate="epoch %{x}<br>agreement %{y:.4f}<extra></extra>",
                    ),
                    row=2, col=col,
                )
        add_reference_line(fig, y=0.0, label="no agreement", row=2, col=col)
        fig.update_xaxes(title_text="Epoch", row=2, col=col)
        fig.update_xaxes(title_text="Epoch", row=1, col=col)
        fig.update_yaxes(title_text="Masked MSE" if col == 1 else "", row=1, col=col)
        fig.update_yaxes(
            title_text="Pearson, held-out perturbations" if col == 1 else "", row=2, col=col
        )
    apply_theme(fig, height=620)
    save_figure_bundle(
        fig, "training_curves",
        data={"epochs": epochs.to_dict(orient="records"), "runs": runs.to_dict(orient="records")},
        root=paths.repo_dir() / "figures",
        alt=(
            "Validation loss per epoch for the ten fine-tuning runs, and beneath it the "
            "agreement between the predicted and the measured perturbation response on the "
            "same validation perturbations, which the recipe records but does not select on."
        ),
    )
    return {"fig": fig, "n_runs": int(runs.shape[0]) if not runs.empty else 0}


def build_all(summary: Path) -> dict:
    summary = Path(summary)
    restricted = pd.read_parquet(summary / "restricted.parquet")
    band = pd.read_parquet(summary / "null_band.parquet")
    direct = (
        pd.read_parquet(summary / "direct.parquet")
        if (summary / "direct.parquet").exists()
        else pd.DataFrame()
    )
    epochs = (
        pd.read_parquet(summary / "training_epochs.parquet")
        if (summary / "training_epochs.parquet").exists()
        else pd.DataFrame()
    )
    runs = (
        pd.read_parquet(summary / "training_runs.parquet")
        if (summary / "training_runs.parquet").exists()
        else pd.DataFrame()
    )
    out = paths.repo_dir() / "figures"
    return {
        "swap_before_after": figure_swap(restricted, band, out),
        "predicted_and_features": figure_predict_vs_features(restricted, direct, band, out),
        "training_curves": figure_training(epochs, runs, out),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/summary")
    a = ap.parse_args()
    made = {
        name: {k: v for k, v in res.items() if k != "fig"}
        for name, res in build_all(Path(a.summary)).items()
    }
    print(json.dumps(made, indent=2, default=str)[:2000])
