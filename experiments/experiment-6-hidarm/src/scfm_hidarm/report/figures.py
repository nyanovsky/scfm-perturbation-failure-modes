"""The depth profile: held-out score against layer, one curve per model.

    PYTHONPATH=src python -m scfm_hidarm.report.figures --summary results/summary

This is the figure run 6 exists to draw. Run 5 found the perturbed gene's
identity decodable from the other tokens at every depth; run 6 asks whether the
same hidden states resolve *which response* at pair resolution, and the shape of
the answer against depth is the thing that distinguishes the two readings the
design named in advance: **rising then collapsing** is information the head
discards, **flat and low at every depth** is information that is not there.

Decisions this module takes, each of which changes what the panel can be read
to say:

**No point is drawn without its band.** ``hidperm`` --- the real features with
the pairing destroyed and the marginals, the basis, the per-slot scales and the
dose axis all preserved --- is drawn at every depth as a shaded floor, because
a score without its null is a shape rather than evidence. ``hidrand`` (width
only) is drawn as a second, lighter line: at ``k = 64`` the arm is 640 columns
wide and is width-matched to **no** run 3 reference, so the width control is the
only thing that says the number is not a capacity artefact.

**Both translators are drawn, in the same panel, as solid and dashed.** 640
dimensions gives run 3's fixed MLP far more capacity than the 160 the original
``k = 16`` pre-registration would have given it, so the ridge/MLP contrast is
doing real work at this rung and ``mlp`` alone would under-report it.

**Both metric cells get their own figure, and neither confirms the other.** The
verdict is the pre-registered ``residual__weighted__disattenuated``; arms are
only *ordered* by the post-hoc ``centered__weighted__disattenuated``. Drawing
one and calling it "the score" would let a reader take the ordering for the
verdict, which is the error ``docs/INDEX.md`` s5 records as having already
reversed once.

**Captured energy is on the panel, not in a caption.** The fitted basis
under-spans the scored perturbations' subspace unevenly across arms --- 0.859 to
0.9998 at ``k = 64`` --- and ``scgpt_ft`` against ``scgpt_ft_frozen`` is exactly
where the spread is worst, so a fine-tuned-minus-frozen difference read off this
figure is partly a difference in basis coverage. The value travels in the hover
text and in ``data.json`` for every point, and cells below
``BASIS_COVERAGE_TARGET`` are marked, so the caveat cannot be separated from the
number it qualifies.

**A missing cell is drawn as missing.** A model, line or layer that was
requested and not scored is named in the figure's data and raises rather than
producing a panel with a shorter line: a curve that is short because a fit
failed looks exactly like a curve that is short because the model has fewer
layers.

Why cell lines are columns rather than separate figures: they are replicates of
the same question, not levels of it, which is run 5's argument reused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle

from .. import config as C
from .. import paths

#: guide lines are a neutral that is not an entity colour, so nothing meaning
#: "a model" shares a colour with something meaning "a reference". Run 5's value.
GUIDE = "#A8A399"

#: How a model is named on a panel. The two fine-tuned keys are named by what
#: they are --- run 4's own recipe, table trainable or frozen --- rather than by
#: their config key, because "scgpt_ft" on an axis does not say which arm.
MODEL_LABEL = {
    "scgpt": "scGPT",
    "scprint": "scPRINT",
    "transcriptformer": "TranscriptFormer",
    "scgpt_ft": "scGPT fine-tuned (own table, trainable)",
    "scgpt_ft_frozen": "scGPT fine-tuned (own table, frozen)",
}
#: the entity whose colour each model key borrows. Run 3's palette holds one
#: colour per model and the fine-tuned keys are the same model, so they reuse
#: scGPT's colour and are separated by dash pattern --- the rule README s3
#: records for run 6, because ``EDITORIAL_8`` was exhausted before run 6 existed.
MODEL_ENTITY = {
    "scgpt": "scgpt",
    "scprint": "scprint",
    "transcriptformer": "transcriptformer",
    "scgpt_ft": "scgpt",
    "scgpt_ft_frozen": "scgpt",
}
#: how the three scGPT-coloured curves are told apart
MODEL_DASH = {
    "scgpt": "solid",
    "scprint": "solid",
    "transcriptformer": "solid",
    "scgpt_ft": "dash",
    "scgpt_ft_frozen": "dot",
}
LINE_LABEL = {"hepg2": "HepG2", "jurkat": "Jurkat"}

#: The reference arms drawn on the panel as horizontal guides, in the same cell
#: and translator as the curves. Not decoration: the claims discipline is that a
#: score is quoted against its reference arm and its null band and NEVER bare,
#: and run 6's absolute numbers are small on the noise-ceiling scale --- run 3's
#: best arm reaches 0.2719 (HepG2) / 0.2371 (Jurkat) --- so a curve at 0.14 read
#: without them says nothing about whether the hidden state is any good.
#:
#: Two are drawn and not six. ``coexp`` is the floor the run exists to beat ---
#: "does a foundation model know anything ordinary co-expression does not" --- and
#: ``scgpt_emb`` is the ceiling that matters, the STATIC gene table that run 3
#: found beat every queried response feature. The others (``coexp_node``,
#: ``scprint_emb``, ``transcriptformer_emb``, ``null_randpair``) are in
#: ``<sweep>_references.csv`` and in this figure's ``data.json`` --- a panel with
#: six guide lines on it is a panel nobody reads.
REFERENCE_GUIDES = ("coexp", "scgpt_emb")
REFERENCE_GUIDE_LABEL = {
    "coexp": "co-expression",
    "coexp_node": "co-expression, node only",
    "scgpt_emb": "scGPT gene table (static)",
    "scprint_emb": "scPRINT gene table (static)",
    "transcriptformer_emb": "TranscriptFormer gene table (static)",
    "null_randpair": "null_randpair",
}
#: solid is the pre-registered translator; the other is drawn thinner
TRANSLATOR_WIDTH = {"mlp": 2.6, "ridge": 1.4}
TRANSLATOR_MARKER = {"mlp": "circle", "ridge": "diamond-open"}


def entity_colors() -> dict[str, str]:
    """Run 3's shared entity-colour map, copied into this run's ``figures/``.

    Copied rather than imported for the reason run 5 gives: a *rule* is imported
    so it cannot drift into incomparability, but a *label table* is the opposite
    case --- copying it means a recolouring in run 3 cannot silently rename a
    series here.
    """
    path = paths.figures_dir() / "entity_colors.json"
    if not path.exists():
        raise RuntimeError(
            f"entity_colors.json is missing at {path}; without it the models would be "
            "recoloured relative to run 3's report and the paper"
        )
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def load(summary_dir: Path,
         sweep: str = "stage1") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The tracked depth table, its null bands and the reference arms.

    The references are **required**, not optional: a depth curve drawn without
    them is a bare score, and this run's numbers are small on the noise-ceiling
    scale. A missing file fails rather than producing a panel with no baseline.
    """
    d = Path(summary_dir)
    depth, bands = d / f"{sweep}_depth.csv", d / f"{sweep}_null_bands.csv"
    refs = d / f"{sweep}_references.csv"
    for f in (depth, bands, refs):
        if not f.exists():
            raise RuntimeError(
                f"{f} does not exist; run `python -m scfm_hidarm.analysis --sweep {sweep}` "
                "first. A missing file fails here, it never draws an empty panel"
            )
    return pd.read_csv(depth), pd.read_csv(bands), pd.read_csv(refs)


def _band(fig: go.Figure, x, lo, hi, colour: str, row: int, col: int, name: str) -> None:
    """One shaded null band, drawn under the curves it is the floor for."""
    fig.add_trace(
        go.Scatter(
            x=list(x) + list(x)[::-1],
            y=list(hi) + list(lo)[::-1],
            fill="toself", fillcolor=_rgba(colour, 0.13),
            line={"width": 0}, hoverinfo="skip", showlegend=False, name=name,
        ),
        row=row, col=col,
    )


def depth_profile(
    depth: pd.DataFrame,
    bands: pd.DataFrame,
    refs: pd.DataFrame,
    metric: str,
    out_root: Path,
    name: str,
) -> dict:
    """One figure: layer on x, held-out score on y, one curve per (model, translator).

    Point estimates only --- no bootstrap interval is drawn, and the figure says
    so on its face. Every interval in this run is a **group** bootstrap over
    response groups rather than over perturbations (HepG2's 524 rows fall into
    226 groups, one of them 149 rows), so a row-level bar would be too narrow
    for the design and a group bootstrap is not free. Drawing no bar is honest;
    drawing the cheap one is not.
    """
    lines = [l for l in C.CELL_LINES if l in set(depth["cell_line"])]
    if not lines:
        raise RuntimeError(
            f"the depth table holds cell lines {sorted(set(depth['cell_line']))} and none of "
            f"{list(C.CELL_LINES)}; an empty panel would read as 'nothing cleared its null'"
        )
    colour = entity_colors()
    fig = make_subplots(
        rows=1, cols=len(lines), shared_yaxes=True, horizontal_spacing=0.055,
        subplot_titles=[LINE_LABEL.get(l, l) for l in lines],
    )

    records: list[dict] = []
    reference_rows: list[dict] = []
    seen_models: list[str] = []
    for ci, line in enumerate(lines, start=1):
        block = depth[depth["cell_line"] == line]
        # the machinery null, drawn first so every curve sits on top of its floor
        nb = bands[(bands["cell_line"] == line) & (bands["metric"] == metric)]
        for model, sub in nb.groupby("model"):
            sub = sub.sort_values("layer")
            if sub["hidperm_max"].notna().any():
                _band(fig, sub["layer"], np.zeros(len(sub)), sub["hidperm_max"],
                      colour[MODEL_ENTITY[model]], 1, ci, f"{model} hidperm")

        for model, msub in block[block["variant"] == "hid"].groupby("model"):
            ent = colour[MODEL_ENTITY[model]]
            for translator, tsub in msub.groupby("translator"):
                tsub = tsub.sort_values("layer")
                energy = tsub["captured_energy_ratio"].to_numpy()
                below = ~tsub["captured_energy_meets_target"].to_numpy(dtype=bool)
                fig.add_trace(
                    go.Scatter(
                        x=tsub["layer"], y=tsub[metric],
                        mode="lines+markers",
                        line={"color": ent, "width": TRANSLATOR_WIDTH.get(translator, 2.0),
                              "dash": MODEL_DASH.get(model, "solid")},
                        marker={
                            "symbol": TRANSLATOR_MARKER.get(translator, "circle"),
                            "size": 8,
                            "line": {"color": ent,
                                     "width": np.where(below, 2.4, 0.0).tolist()},
                            "color": np.where(below, "rgba(0,0,0,0)", ent).tolist(),
                        },
                        name=f"{MODEL_LABEL.get(model, model)} · {translator}",
                        legendgroup=model,
                        showlegend=ci == 1,
                        customdata=np.stack([energy, tsub["feature_dim"]], axis=1),
                        hovertemplate=(
                            f"{MODEL_LABEL.get(model, model)} · {translator}<br>"
                            "layer %{x}<br>score %{y:.4f}"
                            "<br>captured energy %{customdata[0]:.4f}"
                            "<br>d = %{customdata[1]}<extra></extra>"
                        ),
                    ),
                    row=1, col=ci,
                )
                if model not in seen_models:
                    seen_models.append(model)
                records += [
                    {"cell_line": line, "model": model, "translator": translator,
                     "layer": int(r.layer), "score": float(getattr(r, metric)),
                     "captured_energy_ratio": float(r.captured_energy_ratio),
                     "captured_energy_meets_target": bool(r.captured_energy_meets_target),
                     "feature_dim": int(r.feature_dim)}
                    for r in tsub.itertuples()
                ]
        fig.add_hline(y=0.0, line={"color": GUIDE, "dash": "dot", "width": 1}, row=1, col=ci)
        # the reference arms, drawn in the SAME cell and the SAME translator as
        # the curves. Read in the pre-registered translator only: coexp under
        # ridge is a different number and putting both on the panel would invite
        # a reader to compare an mlp curve against a ridge baseline.
        rline = refs[(refs["cell_line"] == line) & (refs["translator"] == "mlp")]
        for arm in REFERENCE_GUIDES:
            hit = rline[rline["arm"] == arm]
            if hit.empty:
                raise RuntimeError(
                    f"the reference arm {arm!r} is absent for {line} in the tracked "
                    "references table; a depth panel drawn without its baseline is a bare "
                    "score, which is the one thing this run's claims discipline forbids"
                )
            value = float(hit[metric].iloc[0])
            add_reference_line(
                fig, y=value,
                label=f"{REFERENCE_GUIDE_LABEL.get(arm, arm)} {value:.3f}",
                row=1, col=ci,
            )
            reference_rows.append(
                {"cell_line": line, "arm": arm, "translator": "mlp",
                 "label": REFERENCE_GUIDE_LABEL.get(arm, arm), metric: value}
            )
        fig.update_xaxes(title_text="transformer block", row=1, col=ci,
                         tickmode="array",
                         tickvals=sorted(set(block["layer"].dropna().astype(int))))
    fig.update_yaxes(title_text="held-out score (noise-ceiling scale)", row=1, col=1)

    ordering = metric == C.ORDERING_METRIC
    role = "ordering cell, post-hoc" if ordering else "pre-registered verdict cell"
    # The caption sits BELOW the legend and the bottom margin is sized for both.
    # Plotly places a paper-referenced annotation without reserving room for it,
    # so a caption at a y the margin does not cover is drawn on top of whatever
    # is already there --- which on the first render put three lines of prose
    # through the legend and through both x-axis titles.
    apply_theme(
        fig, height=680,
        title={
            "text": (
                "Hidden-state response arm — held-out score by depth<br>"
                f"<span style='font-size:13px;color:#5A554C'>{metric} · {role} · "
                f"k = {C.K_PRIMARY}, arm dimension {C.N_SLOTS * C.K_PRIMARY}</span>"
            ),
            "x": 0.0, "xanchor": "left", "y": 0.965, "yanchor": "top",
        },
        # ⚠️ xref/yref MUST be set to "paper" here. The house template places the
        # legend at ``yref: "container", y: 0, yanchor: "bottom"`` --- pinned to
        # the very bottom of the image --- and overriding only ``y`` leaves that
        # container reference in place, so y = -0.30 lands BELOW the image and
        # the legend silently disappears. An eight-series figure with no legend
        # renders perfectly and says nothing.
        legend={"orientation": "h", "yref": "paper", "y": -0.24, "yanchor": "top",
                "xref": "paper", "x": 0, "xanchor": "left", "font": {"size": 11}},
        margin={"t": 110, "b": 300, "l": 76, "r": 28},
    )
    fig.add_annotation(
        x=0, y=-0.60, xref="paper", yref="paper", showarrow=False,
        align="left", xanchor="left", yanchor="top",
        font={"size": 11, "color": "#5A554C"},
        text=(
            "Shaded floor: <b>hidperm</b>, the machinery null — the real features with the "
            "pairing destroyed and the marginals,<br>basis, per-slot scales and dose axis "
            "preserved. Drawn at every depth; all 38 cells clear it.<br>"
            "<b>Point estimates only.</b> No interval is drawn: every interval in this run is "
            "a group bootstrap over response<br>groups (HepG2 226, largest 149; Jurkat 452), "
            "so differences of this size are not resolved here.<br>"
            "Hollow markers: the fitted basis captures less than "
            f"{C.BASIS_COVERAGE_TARGET} of the best k = {C.K_PRIMARY} basis's energy on the"
            "<br>scored perturbations. That bias is downward only and can never inflate a "
            "score."
        ),
    )

    alt = (
        f"Held-out {metric} against transformer block, one curve per model and translator, "
        f"one panel per cell line, at k = {C.K_PRIMARY} (arm dimension "
        f"{C.N_SLOTS * C.K_PRIMARY}). Each model's hidperm null is a shaded floor beneath its "
        "curve. Point estimates only."
    )
    data = {
        "metric": metric,
        "metric_role": "ordering (post-hoc secondary)" if ordering else "pre-registered verdict",
        "k": C.K_PRIMARY,
        "arm_dim": C.N_SLOTS * C.K_PRIMARY,
        "intervals": "none drawn; every interval in this run is a group bootstrap over "
                     "response groups and was not run for this figure",
        "points": records,
        "null_bands": bands[bands["metric"] == metric].to_dict("records"),
        "reference_guides_drawn": reference_rows,
        "reference_arms_all": refs[
            ["cell_line", "arm", "translator", "feature_dim", metric]
        ].to_dict("records"),
        "reference_note": (
            "re-runs of rows run 3 published, scored in this job and never merged into the "
            "run table. Two are drawn on the panel; every one is here."
        ),
    }
    target = Path(save_figure_bundle(fig, name, data=data, alt=alt, root=out_root))
    files = {p.name: p.stat().st_size for p in sorted(target.iterdir())}
    print(f"[figures] {name}: {', '.join(f'{k} {v}B' for k, v in files.items())}", flush=True)
    return {"figure": name, "path": str(target), "files": files, "n_points": len(records)}


def build(summary_dir: Path, out_root: Path, sweep: str = "stage1") -> list[dict]:
    """Both metric cells, each in its own bundle. Neither confirms the other."""
    depth, bands, refs = load(summary_dir, sweep)
    out = paths.ensure(Path(out_root))
    return [
        depth_profile(depth, bands, refs, C.PRIMARY_METRIC, out, "depth_profile_residual"),
        depth_profile(depth, bands, refs, C.ORDERING_METRIC, out, "depth_profile_centered"),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=None,
                    help="directory holding <sweep>_depth.csv (default: results/summary)")
    ap.add_argument("--out", type=Path, default=None,
                    help="figure bundle root (default: figures/)")
    ap.add_argument("--sweep", default="stage1")
    a = ap.parse_args(argv)
    built = build(a.summary or paths.summary_dir(), a.out or paths.figures_dir(), a.sweep)
    print(json.dumps(built, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
