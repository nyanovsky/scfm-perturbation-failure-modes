"""A local stand-in for the platform's ``silico-figures`` package.

Runs 1-3 were built on Goodfire's Silico platform, where the house Plotly style
shipped as a local path dependency at ``/opt/silico/plugins/figures``. That path
does not exist off-platform, and every report builder imports this module at
module level, so without it neither run 2's nor run 3's report can be rebuilt at
all -- see ``results/verification/report_rebuild.json``, whose caveat says
exactly this.

**This is a reconstruction, not the original package.** The theme in
``_theme.json`` was not invented: it was extracted verbatim from the Plotly
``layout.template`` embedded in a figure bundle the real package produced
(``experiments/experiment-3-yga5ws/figures/arm_scores/arm_scores.html``), and
the reference-line shape and annotation styling were read off the same tracked
bundles. So colours, fonts, margins, axis styling and guide lines match what the
platform emitted. What is *not* guaranteed to match is anything the real package
did that leaves no trace in a saved figure -- accessibility linting, colourblind
pair checks, static export defaults, or any layout tweak applied conditionally.

Treat figures rebuilt through this module as faithful in style but not
byte-identical to the platform's. The prose of a report does not pass through
here at all: it is read from the results tables, so report *text and numbers*
are unaffected by using this stand-in.

Public surface, which is exactly what the three runs import:

``EDITORIAL_8``        the eight-colour editorial palette (the template colourway)
``PLOTLY_CONFIG``      the config dict handed to ``fig.to_html``
``apply_theme``        apply the house template and a figure height
``add_reference_line`` a dotted guide line with a labelled annotation
``save_figure_bundle`` write ``<root>/<name>/{<name>.html,data.json,alt.txt}``
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

__all__ = [
    "DIVERGING",
    "EDITORIAL_8",
    "PLOTLY_CONFIG",
    "SEQUENTIAL",
    "TEMPLATE_NAME",
    "add_reference_line",
    "apply_theme",
    "save_figure_bundle",
]

_THEME = json.loads((Path(__file__).parent / "_theme.json").read_text())

#: The editorial palette, in template order. ``EDITORIAL_8[7]`` is the near-black
#: the runs use for emphasis lines, and is also the guide-line colour below.
EDITORIAL_8: list[str] = list(_THEME["layout"]["colorway"])

#: The named colorscales, taken from the same extracted template rather than
#: restated, so they cannot drift from what the theme applies by default. Both
#: are verified against bundles the real package produced: ``SEQUENTIAL`` is the
#: colorscale embedded in ``figures/table_similarity/table_similarity.html``,
#: which is the only tracked bundle that names one explicitly.
SEQUENTIAL: list[list[Any]] = [list(stop) for stop in _THEME["layout"]["colorscale"]["sequential"]]
DIVERGING: list[list[Any]] = [list(stop) for stop in _THEME["layout"]["colorscale"]["diverging"]]

#: Handed to ``fig.to_html(config=...)``; read off the tracked bundles.
PLOTLY_CONFIG: dict[str, Any] = {
    "responsive": True,
    "displayModeBar": "hover",
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "filename": "figure", "scale": 2},
}

TEMPLATE_NAME = "silico_editorial"

#: Guide lines and their labels, as the platform drew them.
_GUIDE_COLOR = EDITORIAL_8[7]
_GUIDE_LINE = {"color": _GUIDE_COLOR, "dash": "dot", "width": 2}
_GUIDE_LABEL_BG = "rgba(246, 245, 240, 0.85)"

pio.templates[TEMPLATE_NAME] = go.layout.Template(_THEME)


def apply_theme(fig: go.Figure, height: int | None = None, **layout: Any) -> go.Figure:
    """Apply the house template, and a height if one is given.

    Mutates and returns ``fig``, which is how every call site uses it -- the
    return value is ignored throughout the three runs.
    """
    fig.update_layout(template=TEMPLATE_NAME, **layout)
    if height is not None:
        fig.update_layout(height=height)
    return fig


def add_reference_line(
    fig: go.Figure,
    x: float | None = None,
    y: float | None = None,
    label: str | None = None,
    row: int | str | None = None,
    col: int | str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Draw a dotted guide line across one axis, optionally labelled.

    Exactly one of ``x`` or ``y`` is given. ``row``/``col`` accept the Plotly
    ``"all"`` sentinel, which the runs use to stripe every panel of a subplot
    grid at once.

    The emitted shape spans the *other* axis's domain, so the line covers the
    panel regardless of its data range -- this is what produces the
    ``yref="y domain"`` seen in the tracked bundles.
    """
    if (x is None) == (y is None):
        raise ValueError("add_reference_line takes exactly one of x= or y=")

    annotation: dict[str, Any] | None = None
    if label:
        annotation = {
            "text": label,
            "showarrow": False,
            "font": {"color": _GUIDE_COLOR, "size": 12},
            "bgcolor": _GUIDE_LABEL_BG,
            "borderpad": 2,
        }

    # Plotly resolves row/col to the right axis refs, including "all"; passing
    # them through is what keeps a subplot grid correct.
    where = {}
    if row is not None:
        where["row"] = row
    if col is not None:
        where["col"] = col

    if x is not None:
        if annotation is not None:
            # top of the panel, label hanging to the right of the line
            annotation |= {"xanchor": "left", "yanchor": "top", "xshift": 6}
        fig.add_vline(
            x=x, line=_GUIDE_LINE, opacity=1,
            annotation=annotation, annotation_position="top right",
            **where, **kwargs,
        )
    else:
        if annotation is not None:
            # right of the panel, label sitting on top of the line
            annotation |= {"xanchor": "right", "yanchor": "bottom", "yshift": 6}
        fig.add_hline(
            y=y, line=_GUIDE_LINE, opacity=1,
            annotation=annotation, annotation_position="top right",
            **where, **kwargs,
        )
    return fig


def save_figure_bundle(
    fig: go.Figure,
    name: str,
    data: Any = None,
    alt: str | None = None,
    root: Path | str | None = None,
) -> Path:
    """Write one figure bundle and return its directory.

    A bundle is ``<root>/<name>/`` holding three files:

    ``<name>.html``  a self-contained fragment loading Plotly from the CDN
    ``data.json``    the plot-ready records, so a figure can be replotted
                     without rerunning the analysis that produced it
    ``alt.txt``      the prose description, also set as the graph div's
                     ``aria-label``

    The returned path is what run 2's ``Fig.save`` uses as the directory to
    write its static PDF/PNG exports into.
    """
    target = Path(root or ".") / name
    target.mkdir(parents=True, exist_ok=True)

    fragment = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config=PLOTLY_CONFIG,
        default_height="100%",
        default_width="100%",
    )
    if alt:
        # the chart is an image to a screen reader, and alt.txt is its text
        fragment = fragment.replace(
            '<div id=', f'<div role="img" aria-label="{html.escape(alt, quote=True)}" id=', 1,
        )
    height = fig.layout.height or 480
    (target / f"{name}.html").write_text(
        f'<div style="height:{int(height)}px; width:100%;">{fragment}</div>'
    )

    if data is not None:
        (target / "data.json").write_text(json.dumps(data, indent=2, default=str))
    if alt:
        (target / "alt.txt").write_text(alt.strip() + "\n")
    return target
