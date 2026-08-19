"""Assemble run 2's methods and implementation report from the artifacts themselves.

Every number is read out of a file that ships with the report, so the page cannot
drift from the blocks it describes. The primary output is one self-contained HTML
page with the figures embedded inline; a Markdown copy is written beside it.

    PYTHONPATH=src uv run python -m scfm_pfm.report.build --bundles results/report/bundles
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from silico_figures import PLOTLY_CONFIG

from .. import config as C
from .. import paths
from . import figures as figmod

TITLE = "Response features for four single-cell foundation models over three query anchors"
LINES = figmod.LINES
LABEL = figmod.LABEL
MODELS = figmod.MODELS
MODEL_LABEL = figmod.MODEL_LABEL
ANCHOR_LABEL = figmod.ANCHOR_LABEL


# --------------------------------------------------------------------------- #
# a tiny document model: the same blocks render to HTML and Markdown
# --------------------------------------------------------------------------- #
@dataclass
class Head:
    level: int
    text: str


@dataclass
class Para:
    text: str


@dataclass
class Bullets:
    items: list[str]


@dataclass
class Table:
    header: list[str]
    rows: list[list]


@dataclass
class Code:
    text: str


@dataclass
class FigRef:
    name: str


@dataclass
class Section:
    blocks: list = field(default_factory=list)


def _esc(text: str) -> str:
    return html_mod.escape(str(text))


def _inline(text: str) -> str:
    """`code` and **bold** only; everything else is escaped."""
    out = _esc(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    return out


def fmt(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{digits}g}"


# --------------------------------------------------------------------------- #
# content
# --------------------------------------------------------------------------- #
def load_bundles(bundles_dir: Path) -> dict:
    out = {}
    for path in sorted(bundles_dir.glob("report_bundle*.json")):
        b = json.loads(path.read_text())
        out[(b["model"], b["cell_line"])] = b
    return out


def _first(bundles: dict, key: str, default=None):
    for b in bundles.values():
        v = b["manifest"].get(key)
        if v:
            return v
    return default


def build_sections(bundles: dict, floor: dict, verification: dict | None) -> list[Section]:
    sections: list[Section] = []
    any_b = next(iter(bundles.values())) if bundles else None
    grid = any_b["manifest"]["value_grid_cp10k"] if any_b else list(C.VALUE_GRID_CP10K)

    # ---- 1. what this is
    s = Section()
    s.blocks += [
        Head(1, TITLE),
        Para(
            "Run 2 of the feature-swap thread. For each of scGPT, AIDO.Cell, scPRINT and "
            "TranscriptFormer it extracts the response of every read-out gene to a change in "
            "every perturbed gene, at three query anchors and in four feature forms, together "
            "with the evidence that those numbers are numerically trustworthy. It answers no "
            "scientific question and compares no models: the comparison run swaps the feature "
            "and keeps everything else fixed."
        ),
        Head(2, "What a feature block is"),
        Para(
            "A model's **response** at read-out gene j is the single scalar it emits for that "
            "gene given a whole-cell input vector. A **response difference** is that scalar from "
            "two forward passes that are identical except for the input value of the perturbed "
            "gene i, subtracted at gene j. The sign convention is perturbed minus baseline "
            "throughout, so a positive entry means raising gene i raises the model's prediction "
            "for gene j. Zero-out is the special case where the perturbed value is 0, reported "
            "with that same convention rather than the source paper's reversed one."
        ),
        Bullets(
            [
                "**zero-out** — one value per gene pair, the naive in-silico perturbation.",
                f"**virtual value perturbation** — the response difference at each of "
                f"{len(grid)} virtual values ({', '.join(fmt(v) for v in grid)} CP10K), concatenated.",
                "**gradient trajectory** — the derivative of the read-out gene's output with "
                "respect to the perturbed gene's input, at each of the same values, concatenated.",
                "**gene embeddings** — each model's own vocabulary embedding for the read-out "
                "genes, the pair feature the comparison run uses as its no-response baseline.",
            ]
        ),
    ]
    if bundles:
        rows = []
        for (model, cl), b in sorted(bundles.items()):
            m = b["manifest"]
            rows.append(
                [
                    MODEL_LABEL[model],
                    LABEL[cl],
                    fmt(b["n_readout"]),
                    fmt(b["n_perturbed"]),
                    m["layout"]["supplied_mode"],
                    fmt(m["layout"]["n_supplied_genes"]),
                    m["layout"]["forward_passes_per_query"],
                ]
            )
        s.blocks += [
            Head(2, "What was produced"),
            Table(
                ["model", "cell line", "read-out genes", "perturbed genes", "supplied set",
                 "supplied genes", "passes per query"],
                rows,
            ),
        ]
    sections.append(s)

    # ---- 2. the expression floor
    s = Section()
    dec = floor["decisions"]
    shared = floor["shared_response"]["by_cell_line_stratum_floor"]
    s.blocks += [
        Head(2, "The read-out set, and what the expression floor does to the ceiling"),
        Head(3, "Procedure"),
        Para(
            "Run 1 called a gene expressed at a pooled control level of 0.5 CP10K and measured "
            "every replicate-split ceiling over that set. Ranking genes by absolute log fold "
            "change preferentially selects small denominators, so a low floor can seat the "
            "ceiling on genes the assay barely measured. The ceilings were recomputed at five "
            f"floors ({', '.join(fmt(f) for f in C.FLOOR_GRID_CP10K)} CP10K) from run 1's "
            "persisted split-half vectors, for both cell lines, both split forms and all four "
            "effect strata. The stated rule was to take the lowest floor at which the "
            "top-stratum ceiling stops rising, within a tolerance of "
            f"{C.FLOOR_PLATEAU_TOLERANCE}."
        ),
        Para(
            "Three further measurements ask whether a higher floor buys its ceiling by narrowing "
            "the read-out to the response every perturbation shares: a leave-one-out mean-effect "
            "baseline (predict each perturbation by the average of the others in its stratum, "
            "built from one replicate half and scored against the other), the median correlation "
            "between different perturbations across halves, and the share of the effect matrix's "
            "sum of squares carried by its first singular direction."
        ),
        Head(3, "Findings"),
        FigRef("floor_ceiling"),
        Para(
            "The ceiling never stops rising inside the grid. In HepG2 the top-stratum ceiling "
            f"goes from {fmt(shared['hepg2|S4_highest|0.5']['ceiling'])} at the 0.5 CP10K floor "
            f"to {fmt(shared['hepg2|S4_highest|10']['ceiling'])} at 10 CP10K, and in Jurkat from "
            f"{fmt(shared['jurkat|S4_highest|0.5']['ceiling'])} to "
            f"{fmt(shared['jurkat|S4_highest|10']['ceiling'])}, while the read-out set falls from "
            f"{fmt(shared['hepg2|S4_highest|0.5']['n_genes'])} to "
            f"{fmt(shared['hepg2|S4_highest|10']['n_genes'])} genes in HepG2. Part of run 1's low "
            "ceiling was therefore the floor rather than the biology, but no floor on the grid "
            "satisfies the plateau rule, and every step up costs most of the read-out set."
        ),
        Para(
            "Extraction therefore uses the inclusive floor: "
            f"{fmt(dec['hepg2']['n_readout_genes'])} read-out genes in HepG2 and "
            f"{fmt(dec['jurkat']['n_readout_genes'])} in Jurkat. Every block stores the read-out "
            "genes in a fixed order with each gene's pooled control level, so any floor on this "
            "curve is a column mask applied at scoring time rather than a re-extraction."
        ),
        FigRef("floor_shared_response"),
        Para(
            "The mean-effect baseline barely moves with the floor while the ceiling climbs, so "
            "the extra correlation a higher floor buys is mostly perturbation-specific. The "
            "warning is at the inclusive floor instead: a predictor that emits the stratum's "
            "average response already reaches "
            f"{fmt(100 * shared['hepg2|S4_highest|0.5']['loo_mean_baseline'] / shared['hepg2|S4_highest|0.5']['ceiling'], 3)}% "
            "of the HepG2 top-stratum ceiling and "
            f"{fmt(100 * shared['hepg2|S3|0.5']['loo_mean_baseline'] / shared['hepg2|S3|0.5']['ceiling'], 3)}% "
            "of the S3 ceiling, leaving "
            f"{fmt(shared['hepg2|S4_highest|0.5']['headroom'])} and "
            f"{fmt(shared['hepg2|S3|0.5']['headroom'])} of correlation for everything "
            "perturbation-specific. A score at any floor should be reported as headroom over "
            "that baseline as a fraction of the ceiling's own headroom, not as a raw correlation."
        ),
    ]
    sections.append(s)

    # ---- 3. anchors
    s = Section()
    s.blocks += [
        Head(2, "The three query anchors"),
        Head(3, "Procedure"),
        Para(
            "Every feature is computed at all three anchors and the three are kept separate. "
            "Each anchor is a CP10K profile over the read-out genes; each model converts CP10K "
            "into its own input scale through one recorded monotone map, so a virtual value means "
            "the same biological level for all four."
        ),
    ]
    if any_b:
        anchors = any_b["manifest"]["anchors"]
        s.blocks.append(
            Bullets(
                [f"**{ANCHOR_LABEL.get(a, a)}** — {meta['description']}" for a, meta in anchors.items()]
            )
        )
    s.blocks += [
        Bullets(
            [
                "AIDO.Cell, scPRINT: the input is a raw count and the query is a virtual cell of "
                "10,000 counts, so a CP10K value is the count.",
                "TranscriptFormer: the same, then clipped at 30, which is the model's own "
                "training-time clip; a query above 30 CP10K saturates and that is recorded.",
                "scGPT: the input is a quantile bin index, and the bin edges are built once per "
                "cell line from that line's mean control profile and then held fixed, so two "
                "queries are comparable.",
            ]
        ),
        Head(3, "Findings"),
        FigRef("response_by_value"),
    ]
    sections.append(s)

    # ---- 4. masking
    s = Section()
    s.blocks += [
        Head(2, "The masking regime, separated from the anchor"),
        Head(3, "Procedure"),
        Para(
            "Three of the four models are bidirectional and see the read-out gene's own input "
            "value while predicting that gene's output, so an unmasked response measures only "
            "what survives after the model has been told the answer. The uniform-virtual anchor "
            "removes that information as a side effect, because every gene sits at the same "
            "constant. The real-background anchors restore it, so the two anchors differ along "
            "two axes at once. To separate them, the mean-control anchor is also computed with "
            "the read-out genes masked, at each model's own convention read from its checkpoint."
        ),
    ]
    mask_rows = []
    for (model, cl), b in sorted(bundles.items()):
        if cl != LINES[0]:
            continue
        spec = b["manifest"].get("mask_spec")
        masked = b["manifest"].get("masked", {})
        if spec:
            mask_rows.append(
                [
                    MODEL_LABEL[model],
                    spec["kind"].replace("_", " "),
                    fmt(spec.get("value")) if spec.get("value") is not None else "n/a",
                    fmt(spec.get("rate")),
                    f"{masked.get('cycles', 0)} cycles of {masked.get('n_blocks_per_cycle', 0)} "
                    f"disjoint blocks" if masked.get("ran") else "not run",
                    spec["source"],
                ]
            )
        else:
            mask_rows.append(
                [MODEL_LABEL[model], "none", "n/a", "n/a", "not applicable",
                 masked.get("reason", "")]
            )
    if mask_rows:
        s.blocks.append(
            Table(["model", "mask kind", "mask value", "rate", "coverage", "source"], mask_rows)
        )
    s.blocks += [
        Para(
            "Masking a random subset independently per pass would leave the unluckiest gene "
            "unmasked for about fifty passes at a 15% rate, so the mask is a seeded partition "
            "into disjoint blocks: every read-out gene is masked exactly once per cycle, at the "
            "model's own pretraining rate, and the cycle is repeated so each gene is read masked "
            f"{C.MASK_MIN_COVERAGE} times. The uniform-virtual anchor needs no masked variant "
            "because the read-out gene's own input already carries no information about its true "
            "level. TranscriptFormer needs none at all: it is autoregressive, so a gene never "
            "sees its own value and is natively in the masked regime."
        ),
        Head(3, "Findings"),
        FigRef("masked_regime"),
    ]
    sections.append(s)

    # ---- 5. gradients and step stability
    s = Section()
    s.blocks += [
        Head(2, "Gradient trajectories and the finite-difference step"),
        Head(3, "Procedure"),
        Para(
            "A gradient-trajectory feature is a magnitude, so a step-dependent derivative would "
            "make it meaningless. Forward-mode automatic differentiation is used wherever the "
            "model's layout is one differentiable function of the input; elsewhere the derivative "
            "is a central difference whose step is selected by sweeping "
            f"{len(C.STEP_GRID['scgpt'])} steps on {C.N_STEP_SWEEP_GENES} perturbed genes at a "
            "4 CP10K base value and taking the smallest step that agrees with the next larger "
            f"step at median Pearson at or above {C.STEP_STABLE_PEARSON} and median relative L2 "
            f"difference at or below {C.STEP_STABLE_RELATIVE_L2}. Where no step qualifies, the "
            "block is flagged: its gradient features are usable for direction and ranking, not "
            "magnitude."
        ),
        Head(3, "Findings"),
        FigRef("step_stability"),
    ]
    step_rows = []
    for (model, cl), b in sorted(bundles.items()):
        for anchor, sel in b["manifest"].get("step_selection", {}).items():
            meta = b["manifest"]["anchors"].get(anchor, {})
            step_rows.append(
                [
                    MODEL_LABEL[model],
                    LABEL[cl],
                    ANCHOR_LABEL.get(anchor, anchor),
                    fmt(sel["selected_step"]),
                    sel["verdict"].replace("_", " "),
                    meta.get("gdt_source", ""),
                ]
            )
    if step_rows:
        s.blocks.append(
            Table(["model", "cell line", "anchor", "selected step", "verdict", "gradient source"],
                  step_rows)
        )
    exact_rows = []
    for (model, cl), b in sorted(bundles.items()):
        for anchor, meta in b["manifest"]["anchors"].items():
            chk = meta.get("exact_vs_finite_difference")
            if chk:
                exact_rows.append(
                    [MODEL_LABEL[model], LABEL[cl], ANCHOR_LABEL.get(anchor, anchor),
                     fmt(chk["pearson"], 6), fmt(chk.get("relative_l2")), fmt(chk["step"])]
                )
    if exact_rows:
        s.blocks += [
            Para(
                "Where an exact derivative exists it is the feature, and the central difference "
                "at the selected step is its cross-check:"
            ),
            Table(["model", "cell line", "anchor", "Pearson", "relative L2", "step"], exact_rows),
        ]
    sections.append(s)

    # ---- 6. normalization
    s = Section()
    s.blocks += [
        Head(2, "Normalization, which is not cosmetic here"),
        Head(3, "Procedure"),
        Para(
            "The four models emit values on four incommensurable scales, and two of them end in "
            "a softmax across the genes supplied in the same forward pass. Three consequences are "
            "handled explicitly: the supplied gene set is fixed per model and cell line and "
            "recorded; the cross-gene mean response is measured per query and a recentred variant "
            "is emitted beside the raw one for the softmax models; and every block also carries a "
            "per-query standardized form, z-scored across read-out genes, so the comparison run "
            "can use features that are invariant to each model's output scale."
        ),
        Para(
            "A predicted log fold change is emitted only where a defensible bridge to a linear "
            "expression scale exists: AIDO.Cell's head predicts log1p CP10K, so the bridge is "
            "expm1; scPRINT and TranscriptFormer predict a share of the cell's counts, which is "
            "proportional to CP10K. scGPT's output is a per-cell quantile bin index, which is not "
            "an expression level and has no cell-independent inverse, so the bridge is declined "
            "rather than invented."
        ),
        Head(3, "Findings"),
        FigRef("softmax_coupling"),
        FigRef("feature_scale"),
    ]
    sections.append(s)

    # ---- 7. context width and ordering
    s = Section()
    s.blocks += [Head(2, "Two architectural constraints that shape the query")]
    cc = None
    for cl in LINES:
        b = bundles.get(("scgpt", cl))
        if b and b["manifest"].get("context_comparison"):
            cc = b["manifest"]["context_comparison"]
            break
    if cc:
        s.blocks += [
            Head(3, "scGPT: how wide the supplied set is"),
            Para(
                "scGPT was trained at a maximum sequence length of 1,200 but carries no "
                "positional encoding over genes, so a longer input is architecturally admissible; "
                "run 1 supplied the whole 7,444-gene universe and recorded it as an "
                "extrapolation. Run 2 measured what the width changes, on "
                f"{cc['n_genes']} common perturbed genes at the mean control anchor, comparing "
                f"the expressed read-out width ({fmt(cc['widths']['readout'])} genes), the full "
                f"universe ({fmt(cc['widths']['universe'])}) and tiles of at most "
                f"{fmt(cc['widths']['tiled'])} sharing a fixed 256-gene prefix."
            ),
            Table(
                ["comparison", "median Pearson", "median relative L2 difference"],
                [
                    [k.replace("_vs_", " vs ").replace("_", " "),
                     fmt(v["median_pearson"]), fmt(v["median_relative_l2"])]
                    for k, v in cc["agreement"].items()
                ],
            ),
            FigRef("scgpt_context"),
            Para(
                "The three widths do not agree closely, so the production sweep uses the tiled "
                f"width ({cc['production_mode']}), the one inside scGPT's training length. The "
                "cost is that a tiled query is several forward passes rather than one "
                "differentiable function of the input, so scGPT's production gradients are "
                "central differences at the selected step; the exact forward-mode derivative is "
                "available in the untiled layout and was validated against a central difference "
                "there."
            ),
        ]
    tf = None
    for cl in LINES:
        b = bundles.get(("transcriptformer", cl))
        if b:
            tf = b["manifest"]["layout"]
            break
    if tf:
        s.blocks += [
            Head(3, "TranscriptFormer: ordering"),
            Para(
                "TranscriptFormer is autoregressive with a causal mask, so gene j responds to "
                "gene i only when i precedes it. The perturbed gene is placed at position 0 of "
                "every pass, which makes every read-out gene measurable; the cost is that the "
                "context differs between queries by that one gene. Under a single fixed ordering "
                f"{fmt(100 * float(tf['unmeasurable_fraction_under_fixed_ordering']), 3)}% of "
                "ordered pairs would be structurally unmeasurable, which is a property of the "
                "model rather than of this run. Each pass opens with the same "
                f"{fmt(tf['context_block_genes'])}-gene prefix and its shares are normalized by "
                "that pass's own total, so values from different passes are comparable."
            ),
        ]
    sections.append(s)

    # ---- 8. wiring
    s = Section()
    s.blocks += [
        Head(2, "Wiring checks"),
        Head(3, "Procedure"),
        Para(
            "The failure that would silently invalidate everything is a gene-index misalignment "
            "between a model's vocabulary order and the read-out gene order. The pre-registered "
            "guard reads the response between the lowest positive value on the grid and the "
            "highest and asks that the perturbed gene's own entry be the largest response in the "
            "vector, whatever its sign; the plan's sign version asks instead that the gene's own "
            "predicted output move the expected way. Each is repeated with the perturbed gene's "
            "own read-out column relabelled at random, which gives the statistic its null."
        ),
        Para(
            "The pre-registered guard turns out to have no power for an autoregressive model. "
            "With the perturbed gene placed first, TranscriptFormer never conditions on its own "
            "value, so its own entry is never the largest response and the permuted control is "
            "zero as well: test and null sit together at the floor and the comparison says "
            "nothing. Two statistics that keep their power there were added: the median rank of "
            "the own entry within the response vector, whose null is the middle of the vector, "
            "and the sign rate, whose null is a coin flip. A misalignment would drive all three "
            "statistics to their nulls at once, so one wide separation is enough to rule it out."
        ),
        Head(3, "Findings"),
        FigRef("wiring_checks"),
    ]
    wire_rows = []
    for (model, cl), b in sorted(bundles.items()):
        for r in b.get("alignment", []):
            wire_rows.append(
                [
                    MODEL_LABEL[model],
                    LABEL[cl],
                    ANCHOR_LABEL.get(r["anchor"], r["anchor"]),
                    fmt(r["n_perturbed"]),
                    f"{fmt(r['own_largest'], 3)} ({fmt(r['own_largest_null'], 3)})",
                    f"{fmt(r['self_rank_median'], 4)} ({fmt(r['self_rank_median_null'], 4)})",
                    f"{fmt(r['sign_rate'], 3)} ({fmt(r['sign_rate_null'], 3)})",
                    ", ".join(r["separated_by"]) or "none",
                ]
            )
    if wire_rows:
        s.blocks += [
            Table(
                ["model", "cell line", "anchor", "genes", "own entry largest (null)",
                 "median self-rank (null)", "own output moves as expected (null)",
                 "separated on"],
                wire_rows,
            ),
            Para(
                "Each cell gives the observed statistic with its permuted null in parentheses. "
                "The last column names the statistics that separate widely enough to establish "
                "alignment for that block."
            ),
        ]
    sections.append(s)

    # ---- 9. the file contract
    s = Section()
    s.blocks += [
        Head(2, "What run 3 reads"),
        Para(
            "A feature block is addressable by (model, anchor, form, cell line) without reading "
            "any extraction code:"
        ),
        Code(
            "features/<model>/<cell_line>/\n"
            "  manifest.json                     anchors, grids, steps, mask spec, substitutions,\n"
            "                                    the model's response contract\n"
            "  readout_genes.parquet             read-out order, model positions, control level\n"
            "  perturbed_genes.parquet           perturbed order, stratum, subset flags\n"
            "  gene_embeddings.npz               (read-out genes, d) vocabulary embedding\n"
            "  step_sweep.parquet                adjacent-step agreement at every swept step\n"
            "  step_selection.json               the selected step and the rule that chose it\n"
            "  anchor=<anchor>/zero_out.npz      (perturbed, read-out)\n"
            "  anchor=<anchor>/vvp.npz           (perturbed, values, read-out)\n"
            "  anchor=<anchor>/gdt.npz           (perturbed, base values, read-out)\n"
            "  anchor=<anchor>/query_summary.parquet   per-query diagnostics\n"
            "  anchor=mean_control_masked/       the masked counterpart, where it applies\n"
        ),
        Para(
            "Each npz carries `raw`, `standardized`, `recentred` where the model's head ends in a "
            "softmax, and `log2fc` where a scale bridge exists, plus the perturbed and read-out "
            "gene identifiers and the value grid, so a block is self-describing."
        ),
    ]
    if verification:
        ok = sum(1 for r in verification["checks"] if r["ok"])
        s.blocks += [
            Head(2, "Verification"),
            Para(
                f"`python -m scfm_pfm.verify_features --all` ran "
                f"{len(verification['checks'])} checks over the artifact tree and passed "
                f"{ok} of them. The full output ships with the report."
            ),
        ]
        failed = [r for r in verification["checks"] if not r["ok"]]
        if failed:
            s.blocks.append(
                Table(
                    ["check", "detail"],
                    [[f"{r['section']}/{r['check']}", r["detail"]] for r in failed],
                )
            )
    sections.append(s)
    return sections


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_html(sections: list[Section], figs: dict) -> str:
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_esc(TITLE)}</title>",
        "</head>",
        "<body>",
        "<main>",
    ]
    fignum, first_chart = 0, True
    for sec in sections:
        parts.append("<section>")
        for b in sec.blocks:
            if isinstance(b, Head):
                parts.append(f"<h{b.level}>{_inline(b.text)}</h{b.level}>")
            elif isinstance(b, Para):
                parts.append(f"<p>{_inline(b.text)}</p>")
            elif isinstance(b, Bullets):
                parts.append("<ul>")
                parts += [f"<li>{_inline(i)}</li>" for i in b.items]
                parts.append("</ul>")
            elif isinstance(b, Table):
                parts.append("<table>")
                parts.append("<tr>" + "".join(f"<th>{_inline(h)}</th>" for h in b.header) + "</tr>")
                for r in b.rows:
                    parts.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
                parts.append("</table>")
            elif isinstance(b, Code):
                parts.append(f"<pre><code>{_esc(b.text)}</code></pre>")
            elif isinstance(b, FigRef):
                fig = figs.get(b.name)
                if fig is None:
                    continue
                fignum += 1
                meta = fig.meta
                fragment = fig.figure.to_html(
                    full_html=False,
                    include_plotlyjs="inline" if first_chart else False,
                    config=PLOTLY_CONFIG,
                )
                first_chart = False
                parts.append(f'<section class="figure" data-sources="{_esc(meta.get("sources", ""))}">')
                parts.append(
                    f'<h3 class="figure-title">Figure {fignum}: {_inline(meta.get("title", b.name))}</h3>'
                )
                parts.append(fragment)
                parts.append(f'<p class="figure-description">{_inline(meta.get("description", ""))}</p>')
                parts.append(f'<p class="figure-findings">{_inline(meta.get("findings", ""))}</p>')
                parts.append(
                    f'<p class="figure-description">Static copies: '
                    f'<code>figures/{b.name}/{b.name}.pdf</code> and '
                    f'<code>figures/{b.name}/{b.name}.png</code>.</p>'
                )
                parts.append("</section>")
        parts.append("</section>")
    parts += [
        f"<p><small>Built {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from the "
        f"artifact tree.</small></p>",
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def render_markdown(sections: list[Section], figs: dict) -> str:
    out: list[str] = []
    fignum = 0
    for sec in sections:
        for b in sec.blocks:
            if isinstance(b, Head):
                out.append("#" * b.level + " " + b.text + "\n")
            elif isinstance(b, Para):
                out.append(b.text + "\n")
            elif isinstance(b, Bullets):
                out.append("\n".join(f"- {i}" for i in b.items) + "\n")
            elif isinstance(b, Table):
                out.append(
                    "| " + " | ".join(str(h) for h in b.header) + " |\n|"
                    + "|".join("---" for _ in b.header) + "|\n"
                    + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in b.rows)
                    + "\n"
                )
            elif isinstance(b, Code):
                out.append("```\n" + b.text + "\n```\n")
            elif isinstance(b, FigRef):
                fig = figs.get(b.name)
                if fig is None:
                    continue
                fignum += 1
                meta = fig.meta
                out.append(
                    f"**Figure {fignum}: {meta.get('title', b.name)}** — {meta.get('description', '')} "
                    f"{meta.get('findings', '')} (`figures/{b.name}/`)\n"
                )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundles", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    repo = paths.repo_dir()
    bundles_dir = Path(args.bundles) if args.bundles else repo / "results" / "report" / "bundles"
    out = paths.ensure(Path(args.out) if args.out else repo / "results" / "report")
    bundles = load_bundles(bundles_dir)
    print(f"[report] {len(bundles)} feature blocks", flush=True)

    floor = json.loads((repo / "results" / "floor_sweep" / "floor_decision.json").read_text())
    vpath = repo / "results" / "verification" / "verify_all.json"
    verification = json.loads(vpath.read_text()) if vpath.exists() else None

    figs = figmod.build_all(bundles, repo)
    sections = build_sections(bundles, floor, verification)
    (out / "feature_extraction_report.html").write_text(render_html(sections, figs))
    (out / "feature_extraction_report.md").write_text(render_markdown(sections, figs))
    print(f"[report] wrote {out / 'feature_extraction_report.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
