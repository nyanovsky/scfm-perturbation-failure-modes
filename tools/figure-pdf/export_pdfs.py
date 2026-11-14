"""Export tracked figure bundles to PDF, without rerunning any experiment.

Every figure bundle's ``<name>.html`` embeds the complete Plotly figure -- its
traces, its layout, and the house template -- inside the ``Plotly.newPlot`` call.
That is enough to reconstruct the figure exactly, so a PDF can be produced from
the bundle alone: no effect matrices, no feature blocks, no artifact store, no
GPU, and no report builder.

That independence is the point. Run 1's figure builder reads ``--root``, the
gitignored artifact mirror, so it cannot be re-run off-platform at all; and runs
2 and 3 would have to go through ``report.build``, which regenerates every
bundle and would overwrite the tracked, house-styled HTML. Reading the committed
HTML sidesteps both problems and exports the figures *as the platform rendered
them*, rather than as a local rebuild would.

Only run 2's builder ever wrote PDFs (``report/figures.py`` calls
``write_image``); runs 1 and 3 stop at the interactive HTML. This script is how
the other two get static exports.

Requires a Chrome for Kaleido -- see README.md. Writes ``<name>.pdf`` beside each
``<name>.html`` unless ``--out`` redirects it, and refuses to clobber an existing
PDF unless ``--force`` is given, so the nine PDFs run 2 rendered on the platform
survive a careless invocation.

    python tools/figure-pdf/export_pdfs.py experiments/*/figures/*/*.html
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

#: Labels naming a null draw rather than a real arm. These points are a cloud by
#: design -- the legend already says "null" -- so drawing 30-odd overlapping
#: names over them buries the arms the figure is actually about.
_NULL_LABEL = re.compile(r"shuffled_labels|^null_rand|^,")

#: Figures whose hover layer carries a dimension nothing else encodes, and how to
#: draw it. Ten of the thirty bundles reference ``%{text}`` in their hover, but
#: most are scatters of thousands of gene-level points where no one reads an
#: individual label; only these lose something a reader needs.
#:
#: ``strip_prefix``  drop a redundant leading ``"<group>, "`` -- the groups are
#:                   already separated spatially, and the suffix is what is lost.
#: ``drop_nulls``    blank out labels matching :data:`_NULL_LABEL`.
#:
#: Two figures are deliberately absent. ``transfer_mode_agreement`` has 284
#: labelled points, and ``feature_scale`` stacks its four labels per model on a
#: log axis where the medians land almost on top of each other; neither can be
#: annotated legibly at any font size, so both stay HTML-only. ``feature_scale``
#: loses least by it -- its marker symbol already separates median from 99th
#: percentile, and only the cell line is hover-only.
LABEL_RULES: dict[str, dict] = {
    "split_gap": {"strip_prefix": True, "drop_nulls": True, "position": "auto", "size": 8},
    "metric_check": {"strip_prefix": False, "drop_nulls": True, "position": "auto", "size": 8},
    "topk_overlap_grid": {"strip_prefix": False, "drop_nulls": False, "position": "top center", "size": 9},
}


#: Plotly's typed-array ``dtype`` codes, as numpy little-endian descriptors.
_DTYPES = {
    "f8": "<f8", "f4": "<f4", "i8": "<i8", "i4": "<i4", "i2": "<i2", "i1": "|i1",
    "u8": "<u8", "u4": "<u4", "u2": "<u2", "u1": "|u1",
}


def coordinates(value) -> list[float]:
    """Coordinates as plain numbers, whatever form the bundle stored them in.

    These bundles hold coordinates as base64 typed arrays --
    ``{"dtype": "f8", "bdata": "..."}`` -- which Plotly deliberately keeps
    undecoded, because the renderer accepts them as they are. Anything that
    *reads* the values has to decode them first; treating the dict as a sequence
    silently yields its two keys instead of the points.
    """
    if isinstance(value, dict) and "bdata" in value:
        raw = base64.b64decode(value["bdata"])
        dtype = _DTYPES.get(value.get("dtype", "f8"), "<f8")
        return np.frombuffer(raw, dtype=dtype).tolist()
    if value is None:
        return []
    return [v for v in value]


def draw_hover_labels(fig: go.Figure, rule: dict) -> go.Figure:
    """Promote hover-only ``text`` onto the canvas, so a static export keeps it.

    A PDF has no hover layer, so any dimension carried only by ``hovertemplate``
    silently disappears from the rendered page -- the points are still plotted,
    but nothing says which is which. This draws those labels for the figures in
    :data:`LABEL_RULES`.

    The exported figure therefore differs from the interactive HTML it came
    from, which is a deliberate trade and is recorded in this tool's README.
    """
    # A label sitting to the right of a point near the right edge is clipped by
    # the plot area, so placement is decided per point against the spread of its
    # own subplot: points past the midpoint get their label on the left instead.
    spans: dict[str, tuple[float, float]] = {}
    for trace in fig.data:
        xs = [v for v in coordinates(getattr(trace, "x", None)) if isinstance(v, (int, float))]
        if not xs:
            continue
        axis = getattr(trace, "xaxis", None) or "x"
        lo, hi = spans.get(axis, (min(xs), max(xs)))
        spans[axis] = (min(lo, min(xs)), max(hi, max(xs)))

    for trace in fig.data:
        if getattr(trace, "text", None) is None or trace.mode != "markers":
            continue
        labels = [str(t) for t in trace.text]
        if rule.get("strip_prefix"):
            labels = [t.split(", ", 1)[-1] for t in labels]
        if rule.get("drop_nulls"):
            labels = ["" if _NULL_LABEL.search(t) else t for t in labels]
        if not any(labels):
            continue
        trace.text = labels
        trace.mode = "markers+text"
        position = rule.get("position", "auto")
        if position == "auto":
            lo, hi = spans.get(getattr(trace, "xaxis", None) or "x", (0.0, 1.0))
            mid = lo + (hi - lo) / 2
            trace.textposition = [
                "middle left" if isinstance(x, (int, float)) and x > mid else "middle right"
                for x in coordinates(trace.x)
            ]
        else:
            trace.textposition = position
        trace.textfont = {"size": rule.get("size", 8)}

    # Plotly autoscales to the markers, not to the text drawn beside them, so a
    # label on an extreme point runs past the plot area and is clipped. Widening
    # each x axis makes room; the axes carry no explicit range of their own.
    pad = rule.get("pad", 0.22)
    for axis, (lo, hi) in spans.items():
        name = "xaxis" + (axis[1:] if len(axis) > 1 else "")
        target = getattr(fig.layout, name, None)
        if target is None or target.range is not None:
            continue
        if target.type == "log":
            # a log axis takes its range in log10 units, and widening it in data
            # units silently drives the real data off the canvas
            continue
        span = (hi - lo) or 1.0
        target.range = [lo - span * pad, hi + span * pad]
    return fig


def figure_from_html(path: Path) -> go.Figure:
    """Rebuild the figure a bundle's HTML fragment draws.

    The fragment calls ``Plotly.newPlot(id, data, layout, config)``. Decoding is
    positional rather than by regex because the arguments contain nested braces
    and brackets that no expression matches reliably; ``raw_decode`` stops at the
    end of each JSON value and hands back the offset to continue from.

    The figure is rebuilt through :func:`plotly.io.from_json` rather than
    ``go.Figure(data=..., layout=...)`` because these bundles store coordinates
    as base64 typed arrays (``{"dtype": ..., "bdata": ...}``). The constructor
    passes those through undecoded, so ``len(trace.x)`` reports the number of
    dict keys instead of the number of points -- fine for rendering, quietly
    wrong for anything that inspects the data.
    """
    text = path.read_text()
    start = text.index("Plotly.newPlot(")
    decoder = json.JSONDecoder()
    data, end = decoder.raw_decode(text, text.index("[", start))
    layout, _ = decoder.raw_decode(text, text.index("{", end))
    return pio.from_json(json.dumps({"data": data, "layout": layout}))


def export(
    src: Path, out_dir: Path | None = None, force: bool = False, label: bool = True,
) -> Path | None:
    """Write one bundle's PDF. Returns the path written, or None if skipped."""
    dest = (out_dir / f"{src.stem}.pdf") if out_dir else src.with_suffix(".pdf")
    if dest.exists() and not force:
        print(f"[pdf] skip {dest} (exists; --force to overwrite)", flush=True)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig = figure_from_html(src)
    rule = LABEL_RULES.get(src.stem) if label else None
    if rule:
        draw_hover_labels(fig, rule)
    fig.write_image(str(dest), format="pdf")
    note = "  [hover labels drawn]" if rule else ""
    print(f"[pdf] {dest}  {dest.stat().st_size / 1024:.1f} KB{note}", flush=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("html", nargs="+", help="figure bundle .html files")
    ap.add_argument("--out", default=None, help="write here instead of beside each .html")
    ap.add_argument("--force", action="store_true", help="overwrite an existing PDF")
    ap.add_argument(
        "--no-labels", action="store_true",
        help="export exactly as the HTML draws it, leaving hover-only labels undrawn",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else None
    written = sum(
        export(Path(p), out_dir, args.force, label=not args.no_labels) is not None
        for p in sorted(args.html)
    )
    print(f"[pdf] {written} written, {len(args.html) - written} skipped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
