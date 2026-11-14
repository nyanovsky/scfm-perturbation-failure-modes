"""The three figures run 5 argues from, built from ``results/probe/summary.parquet``.

    PYTHONPATH=src python -m scfm_probe.report.figures --summary results/probe

Why three and not more. The run asks one question in three steps, and each step
is a different y-unit, so each gets its own figure and no figure mixes units it
cannot label:

``propagation_depth``   does the edit leave its own token at all (S1). If this
                        sits at machine epsilon the decoding question is moot,
                        so it is drawn first and read first.
``identity_retrieval``  is *which* gene was edited recoverable from the other
                        tokens, without training (S2) and with it (S3). This is
                        the headline; if one plot survives into the paper it is
                        this one.
``identity_controls``   the four controls that decide how a positive result
                        would have to be *worded* --- C4 static table, C6
                        norm-only, C7 group-collapsed --- beside the full
                        signature. It ships whichever way S2 comes out, because
                        a null still has to be said correctly.

Everything else the sweep produces (top-10, MRR, the four non-zero doses in the
retrieval statistics, the individual-cell anchor's own depth profile) travels in
the bundles' ``data.json`` rather than on a panel. A figure that shows every
column shows nothing.

Why models share axes. Three models is few enough to read as three lines, and
the depth profile *is* the comparison the run exists to make: information that
rises through the stack and collapses at the head is discarded, flat-and-low at
every depth is absent. One panel per model would put that contrast on the
reader's memory instead of on the page. Cell lines are the columns because they
are replicates of the question, not levels of it.

Why the y axes are logarithmic. Chance top-1 is 1/524 in HepG2 and 1/705 in
Jurkat. On a linear axis every retrieval score that matters is pressed against
the floor and a result at twice chance is indistinguishable from a result at
chance --- which is exactly the distinction the figure has to support. The
propagation ratio spans orders of magnitude for the same reason. A log axis
cannot draw a zero, so a score of exactly zero is **clipped to a stated floor and
drawn with a hollow marker**, never dropped: a point that vanishes is a point a
reader cannot audit.

Absence is drawn as absence. ``score`` may be NaN with an ``n_undefined`` count
and a ``detail`` string; that is a measured absence and is annotated in the panel
with its own reason, in the absent model's own colour, never plotted as zero and
never silently filtered out. Conversely, a row the design says should exist and
does not is a **failure**: each builder names the missing rows and raises rather
than emitting an empty panel.

Why ``propagation_depth`` has two rows and two y-units. TranscriptFormer is
autoregressive: it right-shifts both the gene tokens and the counts, and the
counts enter as an attention *score modifier* rather than through the residual
stream, so a gene's own count is never an input at its own position. The
own-position delta is therefore undefined for that model, which means S1's
**ratio has no denominator at all** --- its ``propagation_ratio`` rows come back
NaN with the reason in ``detail``, and C1 is unavailable for the same reason.
The *numerator* is defined: ``propagation_off_target_norm`` is the absolute
off-target ‖Δh‖, and it is the only quantity TranscriptFormer's depth profile can
be read from.

A ratio and an absolute norm are different units and cannot share an axis, and
TranscriptFormer is not droppable --- it is a third of the run. So the figure is
two panel rows: the ratio on top, for the models that have a denominator, with
TranscriptFormer's absence annotated in its own colour where its line would be;
the absolute off-target norm underneath, where all three are drawn. The second
row is *within-model* readable only: ``d`` is 512, 256 and 2048, so a norm is in
each model's own hidden units and the comparable thing is the shape of the
profile with depth, which is what the row's axis title and its panel note say.
Plotting the two statistics on one axis, or dropping the model that only has one
of them, would each turn a measured property of the architecture into an
apparent property of the biology.

Colour is per entity and comes from ``figures/entity_colors.json``, so a model
carries the same colour here as in run 3's report and the paper. ``ENTITY_LABEL``
/ ``ENTITY_ORDER`` / ``GUIDE`` below are **copied** from
``experiment-3-yga5ws/src/scfm_pfm/report/figures.py:31-63`` rather than
imported: run 5 is a sibling of run 3, not a dependent of it, and a copy is what
keeps a palette edit in run 3 from silently changing a figure here. The colour
*values* are the shared JSON file, so the two do not drift on colour.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle

from .. import config, paths

# --------------------------------------------------------------------------- #
# palette and labels --- copied from run 3, deliberately not imported
# --------------------------------------------------------------------------- #

#: Guide lines (chance levels, null ticks, unit separators) are drawn in a light
#: neutral that is not one of the entity colours, so nothing that means "a model"
#: shares a colour with something that means "a reference".
GUIDE = "#A8A399"

#: Copied verbatim from run 3's report module. Run 5 imports runs 2 and 3 for
#: *rules* (the retrieval ranking, the response groups) because a copied rule can
#: drift into incomparability; a label table is the opposite case --- copying it
#: means a relabelling in run 3 cannot silently rename a series here.
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


def entity_colors() -> dict[str, str]:
    """The shared entity-colour map, read the way run 3 reads it.

    The file is run 3's, copied into this run's ``figures/`` so the three models
    keep the exact colours they carry in the paper. Keys beginning with ``_`` are
    the file's own notes, not entities.
    """
    path = paths.figures_dir() / "entity_colors.json"
    if not path.exists():
        raise RuntimeError(
            f"entity_colors.json is missing at {path}; without it the models would "
            "be recoloured relative to run 3's report and the paper."
        )
    raw = json.loads(path.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


COLOR = entity_colors()

#: How a cell line is named on a panel.
LINE_LABEL = {"hepg2": "HepG2", "jurkat": "Jurkat"}

#: How a split is named on a panel, with the plan's identifier, so a caption and
#: the design document can be lined up without a translation step.
SPLIT_LABEL = {
    "split_half": "split-half retrieval (S2, training-free)",
    "heldout_cell": "held-out-cell probe (S3, trained)",
    "self_position": "perturbed gene's own position (C1)",
}

#: S1's two statistics. The ratio is the PER-POSITION form on purpose: the
#: pre-registered Frobenius ratio divides a norm over n_readout-1 positions by a
#: norm at one, so it grows as sqrt(n_readout) and identical propagation reads
#: 9.3% higher in Jurkat (3247 read-out genes) than in HepG2 (2721). This figure
#: puts the two lines on shared axes, so plotting the undivided form here would
#: show a cell-line difference that is an artefact of the gene count. Both forms
#: are in summary.parquet; the undivided one is what the within-line depth
#: profile was pre-registered on.
#: The ratio is dimensionless;
#: the off-target norm is its numerator, in the model's own hidden units. They
#: are separate rows of ``propagation_depth`` and never share an axis --- see the
#: module docstring for why a model can have the second and not the first.
RATIO_STATISTIC = "propagation_ratio_per_position"
NORM_STATISTIC = "propagation_off_target_norm"

#: The S1 rows, top to bottom, with the axis title that carries each one's unit.
#: The ratio is drawn first because it is the quantity the run's question is
#: phrased in; the norm is drawn beneath it because it is what remains readable
#: when the denominator does not exist.
S1_ROWS = (
    (RATIO_STATISTIC,
     "Off-target ‖Δh‖ per position / own-position ‖Δh‖"
     "<br><sub>dimensionless; divided by √(n_readout−1) so the two cell lines "
     "are comparable</sub>"),
    (NORM_STATISTIC,
     "Off-target ‖Δh‖<br><sub>each model in its own hidden units</sub>"),
)

#: Printed once in the norm row. d is 512 (scGPT), 256 (scPRINT) and 2048
#: (TranscriptFormer), so the heights of two models' norms are not a comparison;
#: the shape of each profile with depth is. Saying so on the panel is cheaper
#: than hoping the axis title is read.
NORM_ROW_CAVEAT = "d = 512 / 256 / 2048 — compare shapes, not heights"

#: The columns ``analysis.py`` promises. Checked on load, because every builder
#: below indexes them by name and a renamed column would otherwise surface as an
#: empty panel rather than an error.
SUMMARY_COLUMNS = (
    "model", "cell_line", "anchor", "layer", "value_cp10k", "statistic", "split",
    "score", "lo", "hi", "null_max", "n_perturbations", "n_undefined", "detail",
)

#: The control panel's rows, in drawing order, top to bottom. The label carries
#: the unit because the units differ: four are top-1 retrieval accuracies and the
#: fifth is a linear CKA, drawn below a separator on the same axis.
#:
#: The unit-normalized variant of S2 --- the other half of C6 --- has **no
#: distinct ``statistic`` value in the frozen contract**: ``pool_unit`` is a
#: second array written under the same statistic name, so there is no row this
#: figure could address it by. It is therefore absent rather than guessed at. If
#: ``analysis.py`` later emits one, add its key here and it will be drawn.
CONTROL_STATISTICS = {
    "retrieval_top1": "full Δh, cosine retrieval",
    "norm_only_top1": "‖Δh‖ magnitudes only (C6)",
    "group_collapsed_top1": "hit if same response group (C7)",
    "cka_static_table": "static gene table, linear CKA (C4)",
}

#: The one row above whose unit is not a top-1 accuracy. Drawn below a separator
#: so no reader compares its height with the accuracies' without noticing.
CKA_STATISTIC = "cka_static_table"

#: A log axis has no zero. A score of exactly zero is a real measurement, so it
#: is drawn at ``floor`` and flagged rather than dropped; the floor is this
#: fraction of the smallest meaningful positive quantity on the panel (chance,
#: where there is a chance level).
LOG_FLOOR_FRACTION = 0.25

#: Vertical offset between the three models' marks inside one control category,
#: in category units. 0.22 keeps three marks distinct without letting a category
#: bleed into its neighbour.
CONTROL_OFFSET = 0.22

#: An absence note is prose read at figure size, so it is hard-wrapped rather
#: than left to the renderer: Plotly does not wrap annotation text, and an
#: unwrapped ``detail`` sentence runs out of its panel and across its neighbour.
#: 44 characters is what fits in the left half of one panel of a two-column grid
#: at 11 px, which is where a note can sit without reaching the far end of a
#: rising profile.
ABSENCE_WRAP = 44

#: How many wrapped lines of a ``detail`` string reach the panel. The full string
#: is in ``data.json`` and in the run's summary table; the panel carries enough of
#: it to say what was absent and why, not the whole sentence.
ABSENCE_MAX_LINES = 4

#: Point size of a panel note. Below the theme's 13, because a note is
#: subordinate to the data, but not below 11: at a single-column width the whole
#: figure is scaled by about three quarters, and these notes carry the reason a
#: series is missing --- the one thing on the panel a reader cannot reconstruct.
ABSENCE_FONT_PX = 11

#: Height of one line of annotation text, in pixels, used to stack several notes
#: in one panel corner without measuring the rendered text.
ABSENCE_LINE_PX = 13


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #


def _line_label(line: str) -> str:
    return LINE_LABEL.get(line, line)


def _rgba(hex_color: str, alpha: float) -> str:
    """A hex entity colour as a translucent fill, for a bootstrap band."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _axis_ref(index: int) -> tuple[str, str]:
    """Plotly's domain refs for the ``index``-th subplot, 1-based row-major."""
    suffix = "" if index == 1 else str(index)
    return f"x{suffix} domain", f"y{suffix} domain"


def load_summary(summary_dir: Path) -> pd.DataFrame:
    """The sweep's summary table, with its contract checked on the way in.

    ``value_key`` is added because the value grid is float and joining a figure's
    required grid against a parquet float column on equality is a coin toss;
    every lookup below goes through the rounded key instead.
    """
    path = summary_dir / paths.summary_table().name
    if not path.exists():
        raise RuntimeError(
            f"{path} does not exist; the figures are built from the sweep summary "
            "and there is nothing to draw without it."
        )
    d = pd.read_parquet(path)
    missing = [c for c in SUMMARY_COLUMNS if c not in d.columns]
    if missing:
        raise RuntimeError(
            f"{path} is missing the columns {missing}; the figure builders index "
            "the summary by name and would otherwise draw empty panels."
        )
    return d.assign(value_key=d.value_cp10k.astype(float).round(6))


def _key(value: float) -> float:
    return round(float(value), 6)


def _missing_rows(
    have: set[tuple],
    models: Sequence[str],
    lines: Sequence[str],
    values: Sequence[float],
) -> list[str]:
    """Which (model, line, layer, value) cells of the design have no row."""
    out = []
    for model in models:
        for line in lines:
            for layer in config.sweep_layers(model):
                for value in values:
                    if (model, line, layer, _key(value)) not in have:
                        out.append(f"{model}/{line}/layer={layer}/value={value:g}")
    return out


def _fail_missing(figure: str, what: str, missing: list[str]) -> None:
    if not missing:
        return
    shown = ", ".join(missing[:8])
    more = f" (and {len(missing) - 8} more)" if len(missing) > 8 else ""
    raise RuntimeError(
        f"{figure}: the summary has no {what} for {len(missing)} cells of the "
        f"design --- {shown}{more}. A figure drawn over a hole would read as a "
        "measured null; refusing to draw it."
    )


def _log_floor(chance: float | None, values: Iterable[float]) -> float:
    """Where the log axis bottoms out, and where an exact zero is drawn.

    Tied to the chance level where there is one, because "below chance" is the
    only thing a zero on this axis has to communicate. Falls back to the smallest
    positive value on the panel when there is no chance level (the propagation
    ratio has none --- it is a ratio, not an accuracy).
    """
    if chance is not None and math.isfinite(chance) and chance > 0:
        return chance * LOG_FLOOR_FRACTION
    positive = [v for v in values if np.isfinite(v) and v > 0]
    return min(positive) / 10.0 if positive else 1e-12


def _clip_to_floor(values: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Values as plotted on a log axis, and a mask of which ones were clipped.

    NaN passes through as NaN --- it is an absence and is annotated, not moved.
    """
    y = np.asarray(values, dtype=float)
    clipped = np.isfinite(y) & (y <= floor)
    return np.where(clipped, floor, y), clipped


def _model_legend(fig: go.Figure, models: Sequence[str], row: int = 1, col: int = 1) -> None:
    """One empty marker trace per model: the legend, in a fixed order."""
    for model in ENTITY_ORDER:
        if model not in models:
            continue
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines+markers", name=ENTITY_LABEL[model],
                line=dict(color=COLOR[model], width=2.4),
                marker=dict(color=COLOR[model], size=8), hoverinfo="skip",
            ),
            row=row, col=col,
        )


def _guide_legend(fig: go.Figure, label: str, symbol: str, row: int = 1, col: int = 1) -> None:
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers", name=label,
            marker=dict(color=GUIDE, size=9, symbol=symbol), hoverinfo="skip",
        ),
        row=row, col=col,
    )


def _line_legend(fig: go.Figure, label: str, width: float, opacity: float,
                 row: int = 1, col: int = 1) -> None:
    """A legend entry for something drawn as a line rather than a mark."""
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines", name=label,
            line=dict(color=GUIDE, width=width), opacity=opacity, hoverinfo="skip",
        ),
        row=row, col=col,
    )


def _fill(fig: go.Figure, x, y, colour: str, alpha: float, row: int, col: int) -> None:
    """A filled region --- a bootstrap band or a null strip --- and nothing else.

    ``mode="lines"`` is not decoration: Plotly's default mode for a short scatter
    is ``lines+markers``, so a band left unset draws a marker at every vertex, in
    the next colour of the template's colourway. Those marks are indistinguishable
    from a fourth series, and on the first render of this module they were exactly
    that --- a green and an olive "model" that do not exist.
    """
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", fill="toself", fillcolor=_rgba(colour, alpha),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ),
        row=row, col=col,
    )


def _reference_label(fig: go.Figure, index: int, text: str,
                     x: float | None = None, y: float | None = None) -> None:
    """A guide line's label, placed *on* the line rather than at the panel edge.

    ``add_reference_line``'s own ``label`` hangs the text at the top right of the
    panel. That is the house style and it reads correctly when the line is near
    the top --- but chance here is 1/524 on a log axis, at the very bottom, and a
    label four inches above its line is a label a reader will attach to the wrong
    thing. In the second column it landed off the edge of the figure entirely. So
    the line is drawn by the house helper and only the text is placed here, at
    the line's own coordinate. Positions are given in log units because every
    axis this labels is logarithmic.
    """
    if (x is None) == (y is None):
        raise ValueError("_reference_label takes exactly one of x= or y=")
    xref, yref = _axis_ref(index)
    if y is not None:
        where = dict(x=0.985, y=math.log10(y), xref=xref,
                     yref=yref.replace(" domain", ""),
                     xanchor="right", yanchor="bottom", yshift=3)
    else:
        where = dict(x=math.log10(x), y=0.995, xref=xref.replace(" domain", ""),
                     yref=yref, xanchor="left", yanchor="top", xshift=4)
    fig.add_annotation(
        showarrow=False, text=text, font=dict(size=ABSENCE_FONT_PX, color=GUIDE),
        bgcolor="rgba(246, 245, 240, 0.85)", borderpad=2, **where,
    )


def _wrap(text: str, width: int = ABSENCE_WRAP,
          max_lines: int = ABSENCE_MAX_LINES) -> tuple[str, int]:
    """Panel prose, hard-wrapped, and how many lines it came to.

    Plotly does not wrap annotation text: an unwrapped ``detail`` sentence runs
    off its panel and over its neighbour, which is how a note meant to record an
    absence ends up obscuring the data. Wrapping here, and returning the line
    count, is also what lets several notes stack in one corner without measuring
    the rendered text.
    """
    lines = textwrap.wrap(" ".join(str(text).split()), width=width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,;") + " …"
    return "<br>".join(lines), len(lines)


def _absence_notes(block: pd.DataFrame) -> list[tuple[str, str]]:
    """One note per model whose score is absent, with the measured reason.

    A NaN score is a datum: the model was queried and the statistic was
    undefined. The reason travels in ``detail``, so it is printed on the panel
    rather than paraphrased, and it is printed in the *model's own colour* --- an
    absence belongs to the series it is missing from, and a reader scanning three
    coloured lines for a fourth should find the answer in that colour.

    Only a model whose every score in the block is NaN gets a note. A model with
    some scores and some gaps is drawn with ``connectgaps=False``, which is the
    gap, and annotating it as absent would overstate what is missing.
    """
    # When EVERY model in the panel is absent for the same reason, say it once. Three
    # models each printing the same paragraph into one panel collides into an
    # unreadable block, and a reader learns nothing from the third copy that the
    # first did not tell them. The per-model, per-colour form below is for the case
    # it was built for: one model missing from a panel where the others are drawn.
    absent_models = [m for m, sub in block.groupby("model", sort=False)
                     if not sub.score.notna().any()]
    n_models = block.model.nunique()
    if n_models and len(absent_models) == n_models:
        details = [str(x) for x in block.detail.dropna().unique() if str(x).strip()]
        shared = details[0] if len(details) == 1 else ""
        if shared:
            # (text, colour), matching the per-model branch below. Neutral, not a
            # model colour: this absence belongs to the whole panel, not to a series.
            #
            # Only the first sentence is drawn. The reason a row carries in `detail`
            # is written for the table, where length costs nothing; on a panel it
            # wraps across the gridlines and gets truncated mid-word, which reads as
            # a rendering fault rather than a recorded absence. The full string is in
            # data.json, which is what a claim traces to.
            first = shared.split(". ")[0].rstrip(".")
            return [(first, GUIDE)]

    notes: list[tuple[str, str]] = []
    for model, sub in block.groupby("model", sort=False):
        if sub.score.notna().any():
            continue
        details = [str(x) for x in sub.detail.dropna().unique() if str(x).strip()]
        reason = details[0] if details else "score undefined, no reason recorded"
        undefined = int(pd.to_numeric(sub.n_undefined, errors="coerce").fillna(0).max())
        suffix = f" ({undefined} undefined)" if undefined else ""
        label = ENTITY_LABEL.get(model, model)
        notes.append((f"{label} — not defined: {reason}{suffix}",
                      COLOR.get(model, GUIDE)))
    return notes


def _free_corner(panel: pd.DataFrame) -> str:
    """Which end of the panel's left edge the notes go in.

    The notes sit on the left, so what matters is where the *first* layer's
    points are, not where the profile ends up: a profile that climbs with depth
    starts low and leaves the top-left free, a flat high one leaves the
    bottom-left free. The chosen end is whichever leaves the larger gap between
    the note and the nearest left-edge point, measured on the log axis the panel
    is actually drawn on --- so the notes will not have to be moved by hand when
    the real numbers come in, and a note that lands on a line is not a caption a
    reader can be asked to ignore.
    """
    v = pd.concat([pd.to_numeric(panel[c], errors="coerce")
                   for c in ("score", "lo", "hi")])
    v = v[np.isfinite(v) & (v > 0)]
    edge = panel[panel.layer == panel.layer.min()] if not panel.empty else panel
    edge = pd.to_numeric(edge.score, errors="coerce") if len(edge) else pd.Series(dtype=float)
    edge = edge[np.isfinite(edge) & (edge > 0)]
    if v.empty or edge.empty:
        return "bottom"
    lo, hi = math.log10(float(v.min())), math.log10(float(v.max()))
    if hi <= lo:
        return "bottom"
    where = (np.log10(edge.to_numpy(dtype=float)) - lo) / (hi - lo)
    return "top" if (1.0 - float(where.max())) > float(where.min()) else "bottom"


def _log_range(values: Iterable[float], corner: str,
               cap: float | None = None) -> list[float] | None:
    """A log axis range that covers ``values`` and reserves room for the notes.

    Plotly's automatic range fits the data and nothing else, so a panel note ends
    up drawn over the series it is about. Reserving a band at the notes' end of
    the axis is the fix that does not require guessing where a line will be:
    the data keeps its own share of the panel and the note has a share of its
    own. ``cap`` bounds the top for a quantity that cannot exceed it --- an
    accuracy must not be given an axis that runs past 1.
    """
    v = np.asarray([float(x) for x in values
                    if x is not None and np.isfinite(x) and x > 0], dtype=float)
    if v.size == 0:
        return None
    lo, hi = math.log10(v.min()), math.log10(v.max())
    span = max(hi - lo, 0.4)
    room, edge = max(0.30 * span, 0.35), max(0.06 * span, 0.06)
    top, bottom = (room, edge) if corner == "top" else (edge, room)
    out = [lo - bottom, hi + top]
    if cap is not None:
        out[1] = min(out[1], math.log10(cap))
    return out


def _annotate(
    fig: go.Figure,
    index: int,
    notes: Sequence[tuple[str, str]],
    y: float | None = None,
    x: float = 0.01,
    corner: str = "bottom",
) -> None:
    """Notes stacked in one panel corner, wrapped, small, in their own colour.

    Each note is its own annotation so it can carry its own colour; they are
    stacked away from the corner by the pixel height of the notes already placed,
    because Plotly reports no text metrics and a single annotation could not be
    multi-coloured.
    """
    xref, yref = _axis_ref(index)
    top = corner == "top"
    anchor = "top" if top else "bottom"
    y = (0.98 if top else 0.02) if y is None else y
    shift = 0
    for text, colour in notes:
        wrapped, n_lines = _wrap(text)
        fig.add_annotation(
            x=x, y=y, xref=xref, yref=yref, xanchor="left", yanchor=anchor,
            yshift=-shift if top else shift, showarrow=False, align="left",
            text=wrapped, font=dict(size=ABSENCE_FONT_PX, color=colour),
        )
        shift += n_lines * ABSENCE_LINE_PX + 4


def _records(d: pd.DataFrame) -> list[dict[str, Any]]:
    """Plotted rows as records, ``null_max`` included, for ``claims_manifest``."""
    cols = [c for c in SUMMARY_COLUMNS if c in d.columns]
    return d[cols].replace({np.nan: None}).to_dict("records")


def _export_pdf(target: Path, name: str, fig: go.Figure) -> bool:
    """The bundle's static page. Run 2's idiom: attempt, report the reason, go on.

    A missing renderer is an environment fact, not a scientific one --- the HTML,
    the data and the alt text are already written --- so it prints and does not
    raise. The caller records whether it happened; nothing claims a PDF exists
    because one was requested.
    """
    try:
        fig.write_image(str(target / f"{name}.pdf"), format="pdf")
        return True
    except Exception as exc:  # noqa: BLE001 - the failure is itself the record
        print(f"[figures] PDF export failed for {name}: {type(exc).__name__}: {exc}",
              flush=True)
        return False


def _save(fig: go.Figure, name: str, data: dict, alt: str, out_root: Path) -> dict:
    target = Path(save_figure_bundle(fig, name, data=data, alt=alt, root=out_root))
    pdf = _export_pdf(target, name, fig)
    files = {p.name: p.stat().st_size for p in sorted(target.iterdir())}
    print(f"[figures] {name}: {', '.join(f'{k} {v}B' for k, v in files.items())}",
          flush=True)
    return {"figure": name, "path": str(target), "files": files, "pdf": pdf, "fig": fig}


# --------------------------------------------------------------------------- #
# S1 --- does the edit leave its own token
# --------------------------------------------------------------------------- #


def _depth_trace(
    fig: go.Figure,
    sub: pd.DataFrame,
    colour: str,
    floor: float,
    label: str,
    hover: str,
    row: int,
    col: int,
) -> bool:
    """One model's depth profile in one panel: its bootstrap band, then its line.

    Returns whether anything was drawn. An all-NaN series is an absence, and an
    absence is annotated by the caller rather than drawn as an empty trace: an
    empty trace still claims a hover target and still tells a reader the model
    was plotted and came out flat, which is the one reading it must not support.
    """
    score = sub.score.to_numpy(dtype=float)
    if not np.isfinite(score).any():
        return False
    y, clipped = _clip_to_floor(score, floor)
    lo, _ = _clip_to_floor(sub.lo.to_numpy(dtype=float), floor)
    hi, _ = _clip_to_floor(sub.hi.to_numpy(dtype=float), floor)
    band = np.isfinite(lo) & np.isfinite(hi)
    if band.any():
        xs = sub.layer.to_numpy()[band]
        _fill(fig, np.concatenate([xs, xs[::-1]]),
              np.concatenate([hi[band], lo[band][::-1]]), colour, 0.14, row, col)
    fig.add_trace(
        go.Scatter(
            x=sub.layer, y=y, mode="lines+markers", showlegend=False,
            line=dict(color=colour, width=2.4), connectgaps=False,
            marker=dict(
                color=np.where(clipped, "rgba(0,0,0,0)", colour), size=8,
                line=dict(color=colour, width=1.6),
            ),
            customdata=score,
            hovertemplate=f"{label}<br>layer %{{x}}: {hover}<extra></extra>",
        ),
        row=row, col=col,
    )
    return True


def propagation_depth(
    summary_dir: Path,
    out_root: Path,
    models: Sequence[str] = paths.MODELS,
) -> dict:
    """Does the edit leave its own token at all, and how does that vary with depth.

    Two rows, because S1 has two statistics in two units and one of the three
    models can only supply the second. ``propagation_ratio`` (top) is the norm of
    the delta at every read-out position other than the perturbed gene's, over
    the norm at its own position. ``propagation_off_target_norm`` (bottom) is that
    numerator alone, in the model's own hidden units.

    The ratio is the quantity the run's question is phrased in, and it is read
    before any decoding question because it bounds all of them: an edit that never
    leaves its own token cannot be decoded from anywhere else. But it needs an
    own-position delta, and TranscriptFormer has none --- it right-shifts tokens
    and counts and admits counts only as an attention score modifier, so a gene's
    own count is never an input at its own position. Its ratio rows are NaN with
    that reason in ``detail``, and the panel says so in TranscriptFormer's own
    colour, in the space where its line would have been.

    Hence the second row. The absolute off-target norm is defined for all three
    models and is the only quantity TranscriptFormer's depth profile can be read
    from at all. It is not a cross-model comparison --- ``d`` is 512, 256 and 2048,
    so the heights are in three different units --- and the panel says that too.
    The one thing it does support is the comparison the run exists to make: a
    profile that rises with depth against one that is flat.

    The zero-out dose is solid and the other four are thin, because zero-out is
    the CRISPRi analog and the only dose whose meaning does not depend on the
    perturbed gene's own scale --- the others are drawn to show the profile does
    not depend on the dose, not to be read individually.

    Layer 0 is annotated rather than plotted. It is control C2: the
    post-embedding state, before block 1, where an edit at gene *i* has not yet
    reached gene *j*'s token, so the off-target delta is exactly zero by
    construction. A log axis cannot show a zero and a linear one would give the
    whole panel to a number that carries no information.
    """
    d = load_summary(summary_dir)
    lines = [x for x in paths.CELL_LINES if x in set(d.cell_line)]
    if not lines:
        raise RuntimeError(
            "propagation_depth: the summary names none of "
            f"{list(paths.CELL_LINES)} in its cell_line column."
        )
    s1 = d[d.statistic.isin([RATIO_STATISTIC, NORM_STATISTIC])
           & (d.anchor == config.PRIMARY_ANCHOR)]

    #: layer 0 is control C2, reported as a number and not as a point
    zero_layer = s1[s1.layer == config.EMBEDDING_LAYER]
    s1 = s1[s1.layer != config.EMBEDDING_LAYER]

    for statistic, _ in S1_ROWS:
        block = s1[s1.statistic == statistic]
        dupes = block.duplicated(
            subset=["model", "cell_line", "layer", "value_key"]).sum()
        if dupes:
            raise RuntimeError(
                f"propagation_depth: {dupes} (model, cell line, layer, value) cells "
                f"carry more than one {statistic} row, so the panel would silently "
                f"plot whichever came first. Splits present: "
                f"{sorted(set(block.split))}."
            )
        have = set(zip(block.model, block.cell_line, block.layer, block.value_key))
        _fail_missing(
            "propagation_depth", f"{statistic} row",
            _missing_rows(have, models, lines, config.VALUE_GRID_CP10K),
        )

    #: only the top row's columns are titled; the rows are named by their y axes,
    #: which is where the unit has to be read anyway
    titles = [_line_label(x) for x in lines] + ["" for _ in lines]
    fig = make_subplots(
        rows=len(S1_ROWS), cols=len(lines), subplot_titles=titles,
        horizontal_spacing=0.11, vertical_spacing=0.11,
    )
    others = [v for v in config.VALUE_GRID_CP10K
              if _key(v) != _key(config.PRIMARY_VALUE_CP10K)]
    floors: dict[str, float] = {}
    for r, (statistic, y_title) in enumerate(S1_ROWS, start=1):
        for c, line in enumerate(lines, start=1):
            index = (r - 1) * len(lines) + c
            panel = s1[(s1.statistic == statistic) & (s1.cell_line == line)]
            at_primary = panel[panel.value_key == _key(config.PRIMARY_VALUE_CP10K)]
            floor = _log_floor(None, panel.score.to_numpy(dtype=float))
            floors[f"{statistic}|{line}"] = floor
            for model in models:
                colour = COLOR[model]
                for value in others:
                    sub = (panel[(panel.model == model)
                                 & (panel.value_key == _key(value))]
                           .sort_values("layer"))
                    y, _ = _clip_to_floor(sub.score.to_numpy(dtype=float), floor)
                    if not np.isfinite(y).any():
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=sub.layer, y=y, mode="lines", showlegend=False,
                            line=dict(color=colour, width=0.9), opacity=0.28,
                            connectgaps=False, hoverinfo="skip",
                        ),
                        row=r, col=c,
                    )
                _depth_trace(
                    fig, (panel[(panel.model == model)
                                & (panel.value_key
                                   == _key(config.PRIMARY_VALUE_CP10K))]
                          .sort_values("layer")),
                    colour, floor, ENTITY_LABEL[model],
                    "%{customdata:.3g}", r, c,
                )
            notes = _absence_notes(at_primary)
            zero = zero_layer[(zero_layer.statistic == statistic)
                              & (zero_layer.cell_line == line)]
            observed = (pd.to_numeric(zero.score, errors="coerce").abs().max()
                        if not zero.empty else np.nan)
            c2 = ("control C2 — layer 0 is 0 by construction"
                  + (f", observed max {observed:.3g}" if np.isfinite(observed) else "")
                  + ", so not plottable here")
            notes.append((c2, GUIDE))
            if statistic == NORM_STATISTIC and c == 1:
                notes.append((NORM_ROW_CAVEAT, GUIDE))
            corner = _free_corner(at_primary)
            _annotate(fig, index, notes, corner=corner)
            fig.update_xaxes(
                title_text=("Transformer block (layer)"
                            if r == len(S1_ROWS) else None),
                tickmode="array", tickvals=sorted(set(panel.layer)), row=r, col=c,
            )
            fig.update_yaxes(
                type="log", row=r, col=c,
                range=_log_range(pd.concat([panel.score, panel.lo, panel.hi]),
                                 corner),
                title_text=y_title if c == 1 else None,
            )
    _model_legend(fig, models)
    _line_legend(fig, f"other doses ({', '.join(f'{v:g}' for v in others)} CP10K)",
                 width=1.4, opacity=0.35)
    apply_theme(fig, height=720, legend=dict(title_text=None))
    alt = (
        "Four panels in two rows, HepG2 on the left and Jurkat on the right. The "
        "top row is the off-target hidden-state change divided by the change at "
        "the perturbed gene's own position, a dimensionless ratio, drawn for "
        "scGPT and scPRINT; TranscriptFormer has no own-position value, so its "
        "ratio is undefined and the panel says so in its colour instead of "
        "drawing a line. The bottom row is the absolute off-target norm, which "
        "all three models have, each in its own hidden units, so profiles are "
        "compared by shape and not by height. Both rows plot against transformer "
        "block on a logarithmic vertical axis: a solid line with a "
        "group-bootstrap band for the zero-out query and four thin lines for the "
        "other doses. Layer 0 is exactly zero by construction and is stated in "
        "each panel rather than plotted, and hollow markers are values clipped "
        "to the axis floor."
    )
    plotted = pd.concat([s1, zero_layer], ignore_index=True)
    data = {
        "records": _records(plotted),
        "statistics": {
            RATIO_STATISTIC: "off-target ‖Δh‖ / own-position ‖Δh‖, dimensionless",
            NORM_STATISTIC: ("off-target ‖Δh‖ alone, in the model's own hidden "
                             "units; the ratio's numerator, and the only S1 "
                             "quantity defined for a model with no own-position "
                             "delta"),
        },
        "d_model": {m: config.D_MODEL[m] for m in models if m in config.D_MODEL},
        "anchor": config.PRIMARY_ANCHOR,
        "primary_value_cp10k": config.PRIMARY_VALUE_CP10K,
        "value_grid_cp10k": list(config.VALUE_GRID_CP10K),
        "embedding_layer_is_control": config.EMBEDDING_LAYER,
        "log_floor_by_panel": floors,
        "models": list(models),
        "sweep_layers": {m: list(config.sweep_layers(m)) for m in models},
    }
    return _save(fig, "propagation_depth", data, alt, out_root)


# --------------------------------------------------------------------------- #
# S2 + S3 --- is the perturbed gene's identity recoverable
# --------------------------------------------------------------------------- #

#: The two splits the headline figure stacks, top row first. Split-half is the
#: primary because ``mean_control`` supplies exactly one delta per perturbation
#: and the only held-out axis it has is over read-out positions; the held-out
#: cell probe is what supports the word "generalizes" and is drawn beneath it.
RETRIEVAL_SPLITS = ("split_half", "heldout_cell")


def identity_retrieval(
    summary_dir: Path,
    out_root: Path,
    models: Sequence[str] = paths.MODELS,
) -> dict:
    """The headline: top-1 retrieval of the perturbed gene, by depth.

    Rows are the two splits, columns the two cell lines, so a model's depth
    profile can be read across and its training-free and trained versions read
    down. Chance is ``1 / n_perturbations``, drawn per column from the summary's
    own count rather than from a remembered 524 / 705.

    The shuffled-label band is drawn once per panel, as the largest
    ``null_max`` in it, and shaded from the axis floor up to that value. One
    strip rather than three keeps the null visually subordinate to the data,
    which is the right weight for a reference; every model's own ``null_max``,
    at every layer, is in ``data.json``, because the claims rule is that an arm
    clears *its own* null in *its own* cell and a panel-level strip cannot
    settle that.

    The held-out-cell row is allowed to be short. That condition costs 24 forward
    passes per perturbation and the plan runs it on scGPT and scPRINT first,
    extending to TranscriptFormer only if the split-half result warrants it, so a
    model absent there is annotated as not run. The split-half row is not allowed
    to be short: every model, line and layer must be present or this raises.
    """
    d = load_summary(summary_dir)
    lines = [x for x in paths.CELL_LINES if x in set(d.cell_line)]
    if not lines:
        raise RuntimeError(
            "identity_retrieval: the summary names none of "
            f"{list(paths.CELL_LINES)} in its cell_line column."
        )
    top1 = d[(d.statistic == "retrieval_top1")
             & (d.value_key == _key(config.PRIMARY_VALUE_CP10K))]

    primary = top1[top1.split == "split_half"]
    have = set(zip(primary.model, primary.cell_line, primary.layer, primary.value_key))
    _fail_missing(
        "identity_retrieval", "split-half retrieval_top1 row",
        _missing_rows(have, models, lines, [config.PRIMARY_VALUE_CP10K]),
    )
    trained = top1[top1.split == "heldout_cell"]
    if trained.empty:
        raise RuntimeError(
            "identity_retrieval: the summary carries no rows with split == "
            "'heldout_cell', so the figure's second row would be empty. Run the "
            "individual-cell anchor, or build the figure from a summary that "
            "records why it was not run."
        )

    #: chance is read from the design's own perturbation count, per cell line
    chance: dict[str, float] = {}
    for line in lines:
        counts = pd.to_numeric(primary[primary.cell_line == line].n_perturbations,
                               errors="coerce").dropna()
        if counts.empty:
            raise RuntimeError(
                f"identity_retrieval: no n_perturbations recorded for {line}, so the "
                "chance level cannot be drawn and the panel would have no reference."
            )
        chance[line] = 1.0 / float(counts.max())

    titles = [f"{_line_label(line)} — {SPLIT_LABEL[split]}"
              for split in RETRIEVAL_SPLITS for line in lines]
    fig = make_subplots(
        rows=len(RETRIEVAL_SPLITS), cols=len(lines), subplot_titles=titles,
        horizontal_spacing=0.10, vertical_spacing=0.14,
    )
    floors: dict[str, float] = {}
    strips: dict[str, float] = {}
    for r, split in enumerate(RETRIEVAL_SPLITS, start=1):
        for c, line in enumerate(lines, start=1):
            index = (r - 1) * len(lines) + c
            panel = top1[(top1.split == split) & (top1.cell_line == line)]
            floor = _log_floor(chance[line], panel.score.to_numpy(dtype=float))
            floors[f"{split}|{line}"] = floor
            top = pd.to_numeric(panel.null_max, errors="coerce").max()
            layers = sorted(set(panel.layer)) or sorted(set(primary.layer))
            if np.isfinite(top) and top > floor:
                strips[f"{split}|{line}"] = float(top)
                _fill(fig, [layers[0], layers[-1], layers[-1], layers[0]],
                      [floor, floor, top, top], GUIDE, 0.20, r, c)
            for model in models:
                sub = panel[panel.model == model].sort_values("layer")
                if sub.empty:
                    continue
                _depth_trace(fig, sub, COLOR[model], floor, ENTITY_LABEL[model],
                             "top-1 %{customdata:.4g}", r, c)
            #: ``annotation_text=""`` is not decoration either: the house
            #: helper builds an annotation whenever it is given a position,
            #: and with no label Plotly falls back to its own placeholder ---
            #: the first render of this module printed "new text" at the top
            #: right of every panel. The label itself is placed below, on the
            #: line rather than above the panel.
            add_reference_line(fig, y=chance[line], row=r, col=c,
                               annotation_text="")
            _reference_label(fig, index,
                             f"chance, 1/{int(round(1 / chance[line]))}",
                             y=chance[line])
            notes = _absence_notes(panel)
            absent = [ENTITY_LABEL.get(m, m) for m in models
                      if panel[panel.model == m].empty]
            if absent:
                notes.append((f"{', '.join(absent)} — condition not run", GUIDE))
            corner = _free_corner(panel)
            _annotate(fig, index, notes, corner=corner)
            fig.update_xaxes(
                title_text="Transformer block (layer)" if r == len(RETRIEVAL_SPLITS)
                else None,
                tickmode="array", tickvals=layers, row=r, col=c,
            )
            #: the axis stops just above the largest band in the panel rather than
            #: running to 1.0. Every score here is a small multiple of chance, and
            #: three decades of empty axis above them would compress the only part
            #: of the panel a reader has to resolve. The floor, the chance level
            #: and the null strip are in the covered set because all three are
            #: drawn and a range that clipped one would hide a reference.
            covered = pd.concat([
                panel.score, panel.lo, panel.hi,
                pd.Series([floor, chance[line], strips.get(f"{split}|{line}")]),
            ])
            fig.update_yaxes(
                type="log", row=r, col=c,
                range=_log_range(covered, corner, cap=1.0),
                title_text="Top-1 retrieval accuracy" if c == 1 else None,
            )
    _model_legend(fig, models)
    _guide_legend(fig, "shuffled-label band (panel maximum)", "square")
    apply_theme(fig, height=780, legend=dict(title_text=None))
    alt = (
        "Four panels in two rows: split-half retrieval on top, the trained "
        "held-out-cell probe below, HepG2 on the left and Jurkat on the right. "
        "Each panel plots top-1 accuracy at recovering which gene was perturbed "
        "against transformer block, on a logarithmic vertical axis, one line per "
        "model with its group-bootstrap band. A dotted line marks chance, one "
        "over the number of perturbations, and a grey strip along the bottom "
        "spans the largest shuffled-label draw in that panel. Hollow markers are "
        "scores of zero clipped to the axis floor; a model that was not run in a "
        "panel, or whose score is undefined, is named there with the reason."
    )
    data = {
        "records": _records(top1),
        "statistic": "retrieval_top1",
        "splits": list(RETRIEVAL_SPLITS),
        "value_cp10k": config.PRIMARY_VALUE_CP10K,
        "chance_by_cell_line": chance,
        "shuffled_label_strip": strips,
        "log_floor_by_panel": floors,
        "models": list(models),
        "note": ("the strip is the largest null_max in its panel; every model's own "
                 "null_max at every layer is in records, which is what an arm must "
                 "be read against"),
    }
    return _save(fig, "identity_retrieval", data, alt, out_root)


# --------------------------------------------------------------------------- #
# C4 / C6 / C7 --- what a positive result would have to be called
# --------------------------------------------------------------------------- #


def _best_layer(primary: pd.DataFrame, model: str, line: str) -> int | None:
    """The layer a model's split-half top-1 peaks at, in one cell line.

    Chosen on the primary statistic and then held fixed for every control, so the
    controls answer "what explains the best case" rather than each being read at
    its own most favourable depth. Choosing per control would guarantee the
    controls look weaker than the thing they are controls for.

    ``None`` when the model has rows but no finite score --- a measured absence,
    which the caller annotates. A *missing row* is the other case and stays
    fatal: no row means the analysis did not run that cell, and drawing the panel
    without it would read as a null. Rows of NaN mean it ran and came out
    undefined, and that is a result with a reason attached.
    """
    sub = primary[(primary.model == model) & (primary.cell_line == line)]
    if sub.empty:
        raise RuntimeError(
            f"identity_controls: {model} in {line} has no split-half retrieval_top1 "
            "row at all, so there is no best layer to read the controls at."
        )
    scored = sub[sub.score.notna()]
    if scored.empty:
        return None
    return int(scored.loc[scored.score.idxmax()].layer)


def identity_controls(
    summary_dir: Path,
    out_root: Path,
    models: Sequence[str] = paths.MODELS,
) -> dict:
    """The controls, at each model's best layer, against its own null.

    This figure decides how the run's result gets *worded*, so it ships whichever
    way the headline comes out. A signature that C4 explains is the paper's
    existing static-table result restated; one that C6's norm-only variant
    matches is the perturbed gene's own expression level, not regulatory reach;
    C7 says whether the strict count is being paid for by paralogs.

    Horizontal, because the category labels carry their units and are long.
    Logarithmic on the score axis, because the categories span the range from
    below chance to a correlation-like CKA and a linear axis would collapse four
    of the five rows onto the origin. On a log axis a bar's *length* is not
    proportional to its value, so the mark is a stem and a point rather than a
    bar --- the point is the datum and the stem only leads the eye to its row.

    The static-table row is a linear CKA and not an accuracy. It is drawn below a
    separator and its label says so, because it shares the axis with four
    accuracies and nothing else on the panel would stop a reader comparing them.
    """
    d = load_summary(summary_dir)
    lines = [x for x in paths.CELL_LINES if x in set(d.cell_line)]
    if not lines:
        raise RuntimeError(
            "identity_controls: the summary names none of "
            f"{list(paths.CELL_LINES)} in its cell_line column."
        )
    at_primary = d[(d.anchor == config.PRIMARY_ANCHOR)
                   & (d.value_key == _key(config.PRIMARY_VALUE_CP10K))]
    primary = at_primary[(at_primary.statistic == "retrieval_top1")
                         & (at_primary.split == "split_half")]
    best = {(m, ln): _best_layer(primary, m, ln) for m in models for ln in lines}

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    #: one note per (model, line) with no defined best layer, carrying the reason
    #: the primary statistic recorded; the controls cannot be read at a layer that
    #: does not exist, and a model dropped without saying so would read as a model
    #: that was never asked
    undefined: dict[str, list[tuple[str, str]]] = {ln: [] for ln in lines}
    for model in models:
        for line in lines:
            layer = best[(model, line)]
            if layer is None:
                undefined[line].extend(
                    _absence_notes(primary[(primary.model == model)
                                           & (primary.cell_line == line)])
                )
                continue
            for stat in CONTROL_STATISTICS:
                sub = at_primary[(at_primary.model == model)
                                 & (at_primary.cell_line == line)
                                 & (at_primary.statistic == stat)
                                 & (at_primary.layer == layer)]
                if len(sub) > 1 and "split_half" in set(sub.split):
                    sub = sub[sub.split == "split_half"]
                if sub.empty:
                    missing.append(f"{model}/{line}/{stat}/layer={layer}")
                    continue
                if len(sub) > 1:
                    raise RuntimeError(
                        f"identity_controls: {model}/{line}/{stat} at layer {layer} "
                        f"has {len(sub)} rows across splits {sorted(set(sub.split))}; "
                        "the panel would plot whichever came first."
                    )
                row = sub.iloc[0]
                rows.append({
                    "model": model, "cell_line": line, "statistic": stat,
                    "layer": layer, "split": row.split,
                    "score": float(row.score) if pd.notna(row.score) else float("nan"),
                    "lo": float(row.lo) if pd.notna(row.lo) else float("nan"),
                    "hi": float(row.hi) if pd.notna(row.hi) else float("nan"),
                    "null_max": float(row.null_max) if pd.notna(row.null_max) else float("nan"),
                    "n_perturbations": (int(row.n_perturbations)
                                        if pd.notna(row.n_perturbations) else None),
                    "n_undefined": (int(row.n_undefined)
                                    if pd.notna(row.n_undefined) else None),
                    "detail": None if pd.isna(row.detail) else str(row.detail),
                })
    _fail_missing("identity_controls", "control row", missing)
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(
            "identity_controls: no model has a finite split-half retrieval_top1 "
            "score in any cell line, so there is no layer any control could be "
            "read at. The reasons are in the summary's detail column; the figure "
            "would be four empty rows and is not drawn."
        )

    order = list(CONTROL_STATISTICS)
    #: y positions run top to bottom in reading order, so index 0 sits highest
    y_of = {stat: float(len(order) - 1 - i) for i, stat in enumerate(order)}
    offsets = {m: (i - (len(models) - 1) / 2) * CONTROL_OFFSET
               for i, m in enumerate(models)}

    fig = make_subplots(
        rows=1, cols=len(lines), subplot_titles=[_line_label(x) for x in lines],
        horizontal_spacing=0.06,
    )
    chance: dict[str, float] = {}
    floors: dict[str, float] = {}
    for k, line in enumerate(lines, start=1):
        panel = table[table.cell_line == line]
        counts = pd.to_numeric(
            primary[primary.cell_line == line].n_perturbations, errors="coerce").dropna()
        chance[line] = 1.0 / float(counts.max()) if not counts.empty else float("nan")
        floor = _log_floor(chance[line] if np.isfinite(chance[line]) else None,
                           panel.score.to_numpy(dtype=float))
        floors[line] = floor
        null_x, null_y = [], []
        for model in models:
            colour = COLOR[model]
            sub = panel[panel.model == model]
            ys = [y_of[s] + offsets[model] for s in sub.statistic]
            score = sub.score.to_numpy(dtype=float)
            x, clipped = _clip_to_floor(score, floor)
            stem_x, stem_y = [], []
            for xi, yi in zip(x, ys):
                if np.isfinite(xi):
                    stem_x += [floor, xi, None]
                    stem_y += [yi, yi, None]
            fig.add_trace(
                go.Scatter(x=stem_x, y=stem_y, mode="lines", showlegend=False,
                           line=dict(color=colour, width=1.3), opacity=0.45,
                           hoverinfo="skip"),
                row=1, col=k,
            )
            fig.add_trace(
                go.Scatter(
                    x=x, y=ys, mode="markers", showlegend=False,
                    marker=dict(
                        color=np.where(clipped, "rgba(0,0,0,0)", colour), size=10,
                        line=dict(color=colour, width=1.8),
                    ),
                    customdata=np.stack([score, sub.layer.to_numpy(dtype=float),
                                         sub.null_max.to_numpy(dtype=float)], axis=-1),
                    hovertemplate=(f"{ENTITY_LABEL[model]}<br>layer %{{customdata[1]:.0f}}"
                                   "<br>score %{customdata[0]:.4g}"
                                   "<br>null max %{customdata[2]:.4g}<extra></extra>"),
                ),
                row=1, col=k,
            )
            for nx, ny in zip(sub.null_max.to_numpy(dtype=float), ys):
                if np.isfinite(nx) and nx > 0:
                    null_x.append(max(nx, floor))
                    null_y.append(ny)
        if null_x:
            fig.add_trace(
                go.Scatter(
                    x=null_x, y=null_y, mode="markers", showlegend=False,
                    marker=dict(color=GUIDE, size=13, symbol="line-ns-open",
                                line=dict(color=GUIDE, width=2)),
                    hovertemplate="shuffled-label maximum %{x:.4g}<extra></extra>",
                ),
                row=1, col=k,
            )
        if np.isfinite(chance[line]):
            add_reference_line(fig, x=chance[line], row=1, col=k,
                               annotation_text="")
            _reference_label(fig, k,
                             f"chance, 1/{int(round(1 / chance[line]))}",
                             x=chance[line])
        #: the unit break: everything above the line is a top-1 accuracy
        fig.add_shape(
            type="line", x0=0, x1=1, xref=f"x{'' if k == 1 else k} domain",
            y0=y_of[CKA_STATISTIC] + 0.5, y1=y_of[CKA_STATISTIC] + 0.5,
            line=dict(color=GUIDE, width=1, dash="dot"), row=1, col=k,
        )
        best_here = " · ".join(
            f"{ENTITY_LABEL[m]} L{best[(m, line)]}" for m in models
            if best[(m, line)] is not None)
        notes = [*undefined[line], *_absence_notes(panel),
                 (f"read at each model's best split-half layer — {best_here}", GUIDE)]
        _annotate(fig, k, notes, y=0.02, x=0.02)
        fig.update_xaxes(
            type="log", row=1, col=k,
            title_text="Top-1 accuracy · CKA (bottom row)",
            range=[math.log10(floor),
                   math.log10(max(1.0, float(np.nanmax(table.score.to_numpy(dtype=float))) * 2))],
        )
        fig.update_yaxes(
            row=1, col=k, tickmode="array",
            tickvals=[y_of[s] for s in order],
            ticktext=[CONTROL_STATISTICS[s] for s in order] if k == 1
            else ["" for _ in order],
            #: the strip below the last category is where the best-layer note
            #: goes; without it the note lands on the CKA row's own marks
            range=[-1.05, len(order) - 0.4], automargin=True,
        )
    _model_legend(fig, models)
    _guide_legend(fig, "shuffled-label maximum", "line-ns-open")
    apply_theme(fig, height=540, legend=dict(title_text=None))
    alt = (
        "Two panels, HepG2 and Jurkat. Five rows of controls read at each model's "
        "best split-half layer: the full hidden-state signature, magnitudes only, "
        "the group-collapsed count, and below a separator the linear CKA against "
        "the model's static gene table, which is not an accuracy. Each model is a "
        "point on a stem on a logarithmic axis, with a short grey tick marking "
        "that point's own largest shuffled-label draw and a dotted line at chance."
    )
    data = {
        "records": table.replace({np.nan: None}).to_dict("records"),
        "statistics": CONTROL_STATISTICS,
        "not_drawn": {
            "unit_normalized_top1": ("the unit-normalized half of C6 has no distinct "
                                     "statistic value in the frozen summary contract"),
        },
        "best_layer": {f"{m}|{ln}": best[(m, ln)] for m in models for ln in lines},
        "no_best_layer": {ln: [t for t, _ in notes] for ln, notes in undefined.items()
                          if notes},
        "anchor": config.PRIMARY_ANCHOR,
        "value_cp10k": config.PRIMARY_VALUE_CP10K,
        "chance_by_cell_line": chance,
        "log_floor_by_cell_line": floors,
        "models": list(models),
    }
    return _save(fig, "identity_controls", data, alt, out_root)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

BUILDERS = {
    "propagation_depth": propagation_depth,
    "identity_retrieval": identity_retrieval,
    "identity_controls": identity_controls,
}


def build_all(
    summary_dir: Path | None = None,
    out_root: Path | None = None,
    models: Sequence[str] = paths.MODELS,
    only: Sequence[str] | None = None,
) -> list[dict]:
    summary_dir = summary_dir or paths.summary_dir()
    out_root = paths.ensure(out_root or paths.figures_dir())
    names = list(only) if only else list(BUILDERS)
    unknown = [n for n in names if n not in BUILDERS]
    if unknown:
        raise RuntimeError(f"no such figure: {unknown}; known: {list(BUILDERS)}")
    return [BUILDERS[name](summary_dir, out_root, models=models) for name in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=None,
                        help="directory holding summary.parquet (default: paths.summary_dir)")
    parser.add_argument("--out", type=Path, default=None,
                        help="figure bundle root (default: paths.figures_dir)")
    parser.add_argument("--models", nargs="+", default=list(paths.MODELS),
                        help="restrict the drawn models; recorded in every data.json")
    parser.add_argument("--figure", nargs="+", default=None, choices=list(BUILDERS),
                        help="build only these figures")
    args = parser.parse_args(argv)

    made = build_all(args.summary, args.out, models=args.models, only=args.figure)
    for entry in made:
        state = "with PDF" if entry["pdf"] else "no PDF"
        print(f"[figures] wrote {entry['figure']} ({len(entry['files'])} files, {state}) "
              f"-> {entry['path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
