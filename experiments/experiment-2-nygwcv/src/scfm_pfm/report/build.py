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
from .. import verify_features as vf
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


def _cell(text: str) -> str:
    """One table cell: no pipes or newlines, which would break the Markdown table."""
    return re.sub(r"\s+", " ", str(text)).replace("|", "/").strip()


def fmt(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{digits}g}"


def _first_line(text, limit: int = 200) -> str:
    """An exception message's first line: the rest is the library's own advice, and a
    hard character cut lands mid-word."""
    line = str(text).splitlines()[0].strip().rstrip(".")
    return line if len(line) <= limit else line[: limit - 1].rsplit(" ", 1)[0] + "..."


def rate(x, digits: int = 3) -> str:
    """A fraction at fixed width, so a column of pass rates lines up."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{float(x):.{digits}f}"


def rank(x) -> str:
    """A position in a vector: an integer where it is one, a half where a median lands
    between two."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    x = float(x)
    return f"{int(x):,}" if x == int(x) else f"{x:,.1f}"


def coverage_tag(bundle: dict) -> str:
    """The perturbed-table tag a block's coverage corresponds to."""
    cov = bundle["manifest"].get("coverage") or {}
    if cov.get("mode") == "full":
        return "full"
    return f"n{cov.get('n_perturbed', bundle['n_perturbed'])}"


def strata_scope(cell_line: str) -> dict | None:
    """How many perturbations the plan's strata hold, before reachability.

    A perturbed gene has to be a column of the supplied set to be perturbed at
    all, so the strata count and the count that can actually be queried are two
    different numbers and the report states both.
    """
    path = paths.handoff_root() / "effects" / cell_line / "perturbations.parquet"
    if not path.exists():
        return None
    perts = pd.read_parquet(path, columns=["target", "effect_stratum"])
    in_strata = perts[perts.effect_stratum.isin(C.PERTURBED_STRATA)]
    return {
        "n_in_strata": int(in_strata.shape[0]),
        "by_stratum": in_strata.effect_stratum.value_counts().to_dict(),
    }


def anchor_levels(
    cell_line: str, clip_cp10k: float | None = None, tag: str | None = None
) -> dict | None:
    """Where the perturbed genes already sit, per anchor, from the shipped tables.

    The response difference vanishes when the queried value equals the gene's own
    level in the anchor, so these numbers are what the size of the response at
    each grid value has to be read against. The perturbed table depends on the
    coverage the block ran at, so the tag selects it: a 250-gene block and a
    full-coverage block do not have the same perturbed genes.
    """
    repo = paths.repo_dir()
    blocks = repo / "results" / "report" / "blocks"
    pg_path = blocks / f"perturbed_genes_{tag}_{cell_line}.parquet"
    if not pg_path.exists():  # run 2's first wave wrote one table per cell line
        pg_path = blocks / f"perturbed_genes_{cell_line}.parquet"
    ro_path = paths.handoff_root() / "readout" / f"readout_genes_{cell_line}.parquet"
    dc_path = paths.demo_cells_dir() / f"demo_cells_{cell_line}.parquet"
    if not (pg_path.exists() and ro_path.exists()):
        return None
    pg = pd.read_parquet(pg_path)
    ro = pd.read_parquet(ro_path)
    cols = pg["readout_col"].to_numpy()
    out = {
        "median_control_cp10k": float(np.median(ro["pooled_control_cp10k"].to_numpy()[cols])),
        "n_perturbed": int(len(pg)),
    }
    if dc_path.exists():
        dc = pd.read_parquet(dc_path)
        merged = ro[["gene_id"]].merge(dc, on="gene_id", how="left")
        cell_cols = [c for c in merged.columns if c.startswith("counts__")]
        totals = np.nansum(dc[cell_cols].to_numpy(float), axis=0)
        cp10k = np.nan_to_num(merged[cell_cols].to_numpy(float)) / totals * 1e4
        subset = cp10k[cols[pg["in_cells_anchor"].to_numpy()]]
        out["cells_median_cp10k"] = float(np.median(subset))
        out["cells_zero_fraction"] = float((subset == 0).mean())
        out["cells_zero_n"] = int((subset == 0).sum())
        # a pair below the read-out floor is delivered to every model as zero, so the
        # 0 CP10K query cannot change it either
        out["cells_below_floor_n"] = int((subset < C.RUN1_FLOOR_CP10K).sum())
        out["cells_n_pairs"] = int(subset.size)
        if clip_cp10k is not None:
            out["cells_above_clip_n"] = int((subset >= clip_cp10k).sum())
            out["cells_clip_cp10k"] = float(clip_cp10k)
    return out


def response_minima(bundles: dict) -> dict:
    """Per anchor, which grid value gives the smallest response in each block."""
    out: dict[str, list] = {}
    for (model, cl), b in sorted(bundles.items()):
        for anchor, queries in b.get("query_summary", {}).items():
            if not queries:
                continue
            best = min(queries, key=lambda r: r["cross_gene_sd"])
            others = [r["cross_gene_sd"] for r in queries
                      if r["value_cp10k"] != best["value_cp10k"]]
            out.setdefault(anchor, []).append(
                {
                    "model": model,
                    "cell_line": cl,
                    "value_cp10k": best["value_cp10k"],
                    "cross_gene_sd": best["cross_gene_sd"],
                    "ratio_to_other_values": (
                        float(np.median(others)) / best["cross_gene_sd"]
                        if best["cross_gene_sd"] else float("nan")
                    ),
                }
            )
    return out


def _join(names: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def pct(x, digits: int = 3) -> str:
    return f"{fmt(100 * float(x), digits)}%"


def _agree(rows: list, key: str) -> str:
    """Describe a column that is usually the same across blocks."""
    values = sorted({r[key] for r in rows})
    if len(values) == 1:
        return fmt(values[0])
    return " and ".join(fmt(v) for v in values)


# --------------------------------------------------------------------------- #
# content
# --------------------------------------------------------------------------- #
def load_bundles(bundles_dir: Path) -> dict:
    out = {}
    for path in sorted(bundles_dir.glob("report_bundle*.json")):
        b = json.loads(path.read_text())
        out[(b["model"], b["cell_line"])] = b
    return out


def load_hidden(repo: Path) -> dict:
    """The visible / masked / median-substituted comparison, where it ran."""
    out = {}
    d = repo / "results" / "report" / "hidden_readout"
    if d.exists():
        for path in sorted(d.glob("*.json")):
            if path.name.startswith("verification_"):
                continue
            rec = json.loads(path.read_text())
            out[(rec["model"], rec["cell_line"])] = rec
    return out


def load_cost_probes(repo: Path) -> dict:
    """Measured throughput and full-coverage projections, per model."""
    out = {}
    d = repo / "results" / "report" / "cost_probe"
    if d.exists():
        for path in sorted(d.glob("cost_probe_*.json")):
            rec = json.loads(path.read_text())
            out[rec["model"]] = rec
    return out


def same_direction(masked_ratio: float, substituted_ratio: float, share: float = 0.5) -> bool:
    """Does the substitution reproduce the mask's effect, and at least half of it?

    Read on a log scale, because a factor of 0.1 and a factor of 10 are the same
    size of effect in opposite directions.
    """
    import math

    lm, ls = math.log(masked_ratio), math.log(substituted_ratio)
    return lm * ls > 0 and abs(ls) >= share * abs(lm)


def _first(bundles: dict, key: str, default=None):
    for b in bundles.values():
        v = b["manifest"].get(key)
        if v:
            return v
    return default


def hidden_blocks(hidden: dict) -> list:
    """The visible / masked / median-substituted comparison, and what it settles."""
    any_rec = next(iter(hidden.values()))
    sub = {cl: None for cl in LINES}
    for (model, cl), rec in hidden.items():
        sub[cl] = rec["substitute_level"]
    rows, verdict_rows = [], []
    for model in MODELS:
        for cl in LINES:
            rec = hidden.get((model, cl))
            if not rec:
                continue
            s_ = rec["summary"]
            r_ = rec["ratios"]
            rows.append(
                [
                    MODEL_LABEL[model],
                    LABEL[cl],
                    fmt(rec["n_perturbed"]),
                    fmt(s_["visible"]["abs_median"], 3),
                    f"{fmt(s_['masked']['abs_median'], 3)} ({rate(r_['masked_over_visible'])})",
                    f"{fmt(s_['median_substituted']['abs_median'], 3)} "
                    f"({rate(r_['median_substituted_over_visible'])})",
                    rate(r_.get("median_substituted_over_masked")),
                ]
            )
            m_, x_ = r_["masked_over_visible"], r_["median_substituted_over_visible"]
            if m_ and x_:
                verdict_rows.append((MODEL_LABEL[model], LABEL[cl], m_, x_, same_direction(m_, x_)))
    survives = sorted({n for n, _, _, _, ok in verdict_rows if ok})
    artefact = sorted({n for n, _, _, _, ok in verdict_rows if not ok})
    both = sorted(set(survives) & set(artefact))
    dilution = sorted(
        {
            round(1.0 - rec["summary"]["masked"]["dilution_factor_vs_run2_estimator"], 3)
            for rec in hidden.values()
        }
    )
    blocks = [
        Head(3, "Separating the mask token from the information it removes"),
        Para(
            "A shrinking response has two readings and the masked block alone cannot tell them "
            "apart. Either the model falls back on what the rest of the cell implies, which is "
            "information removal and a real property of the masked regime, or the mask token is "
            "a value the model is not being asked to read the way pretraining used it, in which "
            "case the whole response is measured off the input distribution and says nothing "
            "about reconstruction. A third condition separates them: instead of the mask token, "
            "the read-out gene's input is set to one uninformative but in-distribution level, "
            "the median pooled-control level across read-out genes ("
            + " and ".join(
                f"{fmt(sub[cl]['value_cp10k'])} CP10K in {LABEL[cl]}"
                for cl in LINES if sub.get(cl)
            )
            + ", at the "
            + " and ".join(
                rate(sub[cl]["quantile_of_readout_levels"]) for cl in LINES if sub.get(cl)
            )
            + " quantile of the read-out genes' own levels). It removes the same gene-specific "
            "information as masking while leaving every input value inside the range the model "
            "was trained on. If the shrinkage survives it, it is information removal; if it only "
            "appears with the mask token, it is an artefact of the query."
        ),
        Para(
            f"All three conditions run in one job over the same {fmt(any_rec['n_perturbed'])} "
            "perturbed genes at the mean-control anchor with one batching, so nothing but the "
            "condition differs. They also correct the masked estimator. The masked read cycles "
            "disjoint blocks and reads each gene only where it is hidden; when the perturbed "
            "gene falls in the same block as the gene being read, the mask overwrites the "
            "perturbation before the forward pass, so that pass is identical to its own baseline "
            "and contributes exactly zero. Run 2's masked block averaged those zeros in, which "
            "shrinks its magnitudes by "
            + " and ".join(pct(d, 3) for d in dilution)
            + " -- one over the number of blocks per cycle. Those passes are skipped here, and "
            "because their contribution is exactly zero the run 2 estimator is reproduced from "
            "these arrays rather than argued about."
        ),
        FigRef("masked_regime"),
        Table(
            ["model", "cell line", "perturbed genes", "visible", "mask token (of visible)",
             "median level (of visible)", "median level over mask token"],
            rows,
        ),
    ]
    verdict = ""
    if survives:
        verdict += (
            f"For {_join(survives)} the substitution moves the response the same way as the mask "
            "and by at least half as much on a log scale, so what the mask takes away is "
            "information: the masked reading stands. "
        )
    if artefact:
        verdict += (
            f"For {_join(artefact)} the substitution does not reproduce what the mask does, so "
            "that model's masked response is at least partly an artefact of being queried "
            "outside its input distribution and must not be read as evidence about "
            "reconstruction. "
        )
    if both:
        verdict += (
            f"{_join(both)} splits between the two cell lines, so its verdict is not a property "
            "of the model alone. "
        )
    blocks.append(
        Para(
            verdict
            + "The rule is stated before the numbers are read: the substitution reproduces the "
            "mask when both ratios sit on the same side of 1 and the substituted one is at least "
            "half the mask's size in log units. Per-gene agreement between the conditions is "
            "recorded beside the medians in each record, because a ratio of medians can hide a "
            "change of pattern."
        )
    )
    agree_rows = []
    for model in MODELS:
        for cl in LINES:
            rec = hidden.get((model, cl))
            if not rec:
                continue
            ag = rec.get("agreement", {})
            agree_rows.append(
                [MODEL_LABEL[model], LABEL[cl]]
                + [
                    rate((ag.get(k) or {}).get("median_pearson_per_gene"), 4)
                    for k in ("visible_vs_masked", "visible_vs_median_substituted",
                              "masked_vs_median_substituted")
                ]
            )
    if agree_rows:
        blocks.append(
            Table(
                ["model", "cell line", "visible vs mask token", "visible vs median level",
                 "mask token vs median level"],
                agree_rows,
            )
        )
        blocks.append(
            Para(
                "Median per-gene Pearson correlation between the two conditions' zero-out "
                "response vectors, over the read-out genes each condition could read. A high "
                "correlation with a small ratio means the same pattern at a smaller size; a low "
                "correlation means the conditions are measuring different things."
            )
        )
    return blocks


def aido_gradient_blocks(bundles: dict, probes: dict) -> list:
    """AIDO.Cell's gradient route, from a measurement rather than a default."""
    probe = (probes.get("aido_cell") or {}).get("per_cell_line", {})
    fm = next(
        (entry["forward_mode_probe"] for entry in probe.values() if "forward_mode_probe" in entry),
        None,
    )
    recorded = sorted(
        {
            (b["manifest"].get("contract", {}) or {}).get("forward_mode_autodiff", "")
            for (model, _), b in bundles.items()
            if model == "aido_cell"
        }
    )
    if fm is None:
        return [
            Para(
                "AIDO.Cell is absent from that table. Its gradient trajectories are central "
                "differences at the selected step, and whether an exact forward-mode derivative "
                "is available for this build was not measured in this run, so no cross-check "
                "stands behind them. Its magnitude evidence is the step sweep alone, which it "
                "passes at the smallest swept step in every anchor and both cell lines."
            )
        ]
    if not fm.get("available"):
        return [
            Para(
                "AIDO.Cell is absent from that table because the exact route does not exist for "
                "this build, which is a measurement rather than an omission. Attempting "
                "forward-mode autodiff through its forward pass raises "
                f"`{_cell(_first_line(fm.get('reason', 'an error')))}`. Its gradient trajectories "
                "are therefore central differences at the selected step with no exact "
                "cross-check behind them, and their magnitude evidence is the step sweep alone, "
                "which they pass at the smallest swept step in every anchor and both cell lines."
            ),
            Para(
                "One record had to be corrected to say this. Run 1 recorded a substitution note "
                "reading `forward-mode AD unavailable (NotImplementedError); used central finite "
                "differences`, while the same contract's `jacobian_available` field asserted the "
                "opposite; that field was an unmeasured default, and it is where run 2's first "
                "report read the claim that the route was available. The contract now carries a "
                "separate `forward_mode_autodiff` field filled from an actual attempt, and every "
                "adapter records why a forward-mode column was unavailable instead of returning "
                "nothing silently."
                + (f" This block records: {_cell(_first_line(recorded[0]))}."
                   if recorded and recorded[0] else "")
            ),
        ]
    two_sided = [r for r in fm.get("per_value", []) if not r["one_sided"]]
    return [
        Para(
            "AIDO.Cell is absent from that table, and the absence is a gap in the extraction "
            "rather than a property of the model: forward-mode autodiff does run through this "
            "build, measured directly rather than inferred. On "
            f"{fmt(fm['n_genes'])} perturbed genes at the mean-control anchor the exact "
            "derivative agrees with the central difference at step "
            f"{fmt(fm['step'])} to a median Pearson of "
            f"{fmt(fm.get('median_pearson_two_sided'), 6)} and a median relative L2 difference of "
            f"{fmt(fm.get('median_relative_l2_two_sided'))} over the "
            f"{len(two_sided)} base values whose difference stays inside the input range. The "
            "shipped gradient trajectories are still the central differences, so this is the "
            "cross-check the block was missing rather than a change to it; an exact-derivative "
            "AIDO.Cell block would be a re-extraction."
        ),
        Para(
            "That also corrects a record. Run 1 recorded a substitution note reading "
            "`forward-mode AD unavailable (NotImplementedError); used central finite "
            "differences`, and the same contract's `jacobian_available` field asserted the "
            "opposite. Neither was a measurement of this build in this environment: the field "
            "was an optimistic default and the note came from an earlier environment. The "
            "contract now carries a separate `forward_mode_autodiff` field filled from an actual "
            "attempt, and every adapter records why a forward-mode column was unavailable rather "
            "than returning nothing silently."
        ),
    ]


def build_sections(
    bundles: dict,
    floor: dict,
    verification: dict | None,
    hidden: dict | None = None,
    probes: dict | None = None,
) -> list[Section]:
    sections: list[Section] = []
    hidden = hidden or {}
    probes = probes or {}
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
            cov = m.get("coverage") or {}
            reach = cov.get("n_in_scope")
            rows.append(
                [
                    MODEL_LABEL[model],
                    LABEL[cl],
                    fmt(b["n_readout"]),
                    f"{fmt(b['n_perturbed'])} of {fmt(reach)}" if reach else fmt(b["n_perturbed"]),
                    "all" if cov.get("mode") == "full" else rate(cov.get("fraction_of_scope")),
                    m["layout"]["supplied_mode"],
                    fmt(m["layout"]["n_supplied_genes"]),
                    m["layout"]["forward_passes_per_query"],
                    f"{round(sum(a.get('seconds', 0.0) for a in m['anchors'].values())):,}",
                ]
            )
        s.blocks += [
            Head(2, "What was produced"),
            Table(
                ["model", "cell line", "read-out genes", "perturbed genes", "coverage",
                 "supplied set", "supplied genes", "passes per query", "GPU seconds, 3 anchors"],
                rows,
            ),
            Para(
                "The last column is the wall clock spent producing the three anchors' zero-out, "
                "virtual-value and gradient features for that block, on one H100. It excludes "
                "the masked block, the step sweep and the layout side checks, and it is the "
                "number to size run 3 against: the four models differ in cost by two orders of "
                "magnitude for the same query."
            ),
        ]
        strata = {cl: strata_scope(cl) for cl in LINES}
        reach = {
            cl: next(
                (
                    b["manifest"]["coverage"]["n_in_scope"]
                    for (m_, c_), b in sorted(bundles.items())
                    if c_ == cl and b["manifest"].get("coverage")
                ),
                None,
            )
            for cl in LINES
        }
        if all(strata.values()) and all(reach.values()):
            s.blocks.append(
                Para(
                    "Two different reductions sit between the plan's effect strata and the "
                    "perturbed-gene column, and they are not the same kind of decision. The plan "
                    f"scoped the {_join(list(C.PERTURBED_STRATA))} strata, which hold "
                    + " and ".join(
                        f"{fmt(strata[cl]['n_in_strata'])} perturbations in {LABEL[cl]}"
                        for cl in LINES
                    )
                    + ". A perturbed gene has to be a column of the supplied set to be perturbed "
                    "at all: the query works by setting that gene's input value, and a gene the "
                    "model was never given has no input to set. Only perturbations whose target "
                    "is itself a read-out gene are therefore reachable, which leaves "
                    + " and ".join(
                        f"{fmt(reach[cl])} in {LABEL[cl]}" for cl in LINES
                    )
                    + ". That reduction is structural and no amount of compute removes it. The "
                    "second is compute: run 2's first wave ran 250 of those per block, sized "
                    "from the slowest model's measured cost, and the three cheaper models now run "
                    "the whole reachable set at every anchor and in the masked variant. Each "
                    "block's manifest records its own coverage and the roll-up fails if one does "
                    "not, because a 250-gene block and a full block share the smaller one's "
                    "perturbed genes but not its row order."
                )
            )
        probe_rows, probe_notes = [], []
        for model, rec in sorted(probes.items()):
            for cl, entry in sorted(rec.get("per_cell_line", {}).items()):
                pr = entry["projection"]
                probe_rows.append(
                    [
                        MODEL_LABEL[model],
                        LABEL[cl],
                        fmt(pr["rate_used_seconds_per_forward"]),
                        fmt(pr["rate_batch_size"]),
                        f"{fmt(pr['best_rate_seconds_per_forward'])} at batch "
                        f"{fmt(pr['best_batch_size'])}",
                        rate(pr["batching_speedup"], 2),
                        f"{fmt(round(pr['observed_subset_three_anchor_seconds'] or 0))} vs "
                        f"{fmt(pr['predicted_subset_three_anchor_seconds'])}",
                        f"{pr['projected_full_three_anchor_seconds'] / 3600:.1f} h",
                        f"{pr['projected_full_everything_seconds'] / 3600:.1f} h",
                    ]
                )
                alt = entry.get("alternate_dtypes", {})
                for dt, info in alt.items():
                    if info.get("best_seconds_per_forward"):
                        probe_notes.append(
                            f"{dt} autocast reaches "
                            f"{fmt(info['best_seconds_per_forward'])} s per forward pass in "
                            f"{LABEL[cl]}"
                        )
        if probe_rows:
            worst = max(probes)
            s.blocks += [
                Head(3, "What full coverage would cost for AIDO.Cell"),
                Para(
                    "AIDO.Cell is the reason the first wave ran a subset, so its extension was "
                    "priced instead of attempted. The projection prices the extraction driver's "
                    "own forward-pass count at a measured rate, and is checked by predicting the "
                    "block that already exists: the observed and predicted three-anchor seconds "
                    "in the table are that check."
                ),
                Table(
                    ["model", "cell line", "s per forward", "at batch", "best measured",
                     "batching speedup", "observed vs predicted, 250 genes",
                     "full, 3 anchors", "full, everything"],
                    probe_rows,
                ),
                Para(
                    "Batching across perturbed genes does not bring this down: the sequence is "
                    f"{fmt(next(iter(probes[worst]['per_cell_line'].values()))['layout']['n_supplied_genes'])} "
                    "genes long, so one forward pass already saturates the GPU and a larger batch "
                    "buys the ratio in the table rather than anything like its size. "
                    + (
                        "Lower precision would: " + _join(probe_notes) + ", against "
                        f"{fmt(probe_rows[0][2])} in float32. That is not a top-up though, "
                        "because it changes the numerics: a block half in float32 and half in "
                        "bfloat16 is not one block. "
                        if probe_notes
                        else ""
                    )
                    + "AIDO.Cell therefore stays at 250 perturbed genes in this run, and the "
                    "number above is what extending it would cost."
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
        Para(
            "That narrow band is a property of the library, not a defect of the measurement. The "
            "source screen perturbs DepMap common-essential genes, which is the study's design "
            "and not a choice of this run: most targets are genes a proliferating cell cannot do "
            "without, so knocking any of them down produces much the same growth-arrest "
            "programme, and the shared part of that programme is most of the transcriptome "
            "response. What is left once the shared signature is predicted for free is "
            f"{fmt(shared['hepg2|S4_highest|0.5']['headroom'])} of correlation in the top "
            f"stratum and {fmt(shared['hepg2|S3|0.5']['headroom'])} in S3, and that band is "
            "the whole space any later comparison between models can resolve. A comparison run "
            "reporting raw correlations on this library will show four models close together and "
            "near the ceiling whatever they have learned; only the headroom over the mean-effect "
            "baseline can separate them, and its size bounds how much separation is available at "
            "all."
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
        uniform = " and ".join(
            f"{fmt(1e4 / b['manifest']['layout']['n_readout_genes'], 5)} CP10K in {LABEL[cl]}"
            for (model, cl), b in sorted(bundles.items())
            if model == sorted({m for m, _ in bundles})[0]
        )
        descriptions = {
            "uniform_virtual": (
                "every supplied gene pinned at the uniform share of a 10,000-count virtual "
                f"cell, {uniform}; the paper's construction, and far off the distribution the "
                "models were trained on"
            )
        }
        s.blocks.append(
            Bullets(
                [
                    f"**{ANCHOR_LABEL.get(a, a)}** — {descriptions.get(a, meta['description'])}"
                    for a, meta in anchors.items()
                ]
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
    minima = response_minima(bundles)
    clip = next(
        (
            b["manifest"]["layout"]["count_clip"]
            for b in bundles.values()
            if b["manifest"]["layout"].get("count_clip") is not None
        ),
        None,
    )
    # the perturbed set now depends on the block's coverage, so the levels a block is
    # read against are looked up by its own coverage tag rather than by cell line alone
    tags = {key: coverage_tag(b) for key, b in bundles.items()}
    levels_by_tag: dict[tuple[str, str], dict | None] = {}
    for (model, cl), tag in tags.items():
        if (tag, cl) not in levels_by_tag:
            levels_by_tag[(tag, cl)] = anchor_levels(cl, clip, tag)
    primary_tag = (
        max(set(tags.values()), key=lambda t: sum(1 for v in tags.values() if v == t))
        if tags
        else "full"
    )
    levels = {cl: levels_by_tag.get((primary_tag, cl)) for cl in LINES}
    if minima:
        parts = []
        uni = minima.get("uniform_virtual", [])
        if uni:
            constants = sorted(
                {round(1e4 / b["manifest"]["layout"]["n_readout_genes"], 4) for b in bundles.values()}
            )
            parts.append(
                f"At the uniform-virtual anchor, where every gene sits at the same constant "
                f"({' and '.join(fmt(c, 5) for c in constants)} CP10K), the "
                f"smallest response is at the {_agree(uni, 'value_cp10k')} CP10K query in all "
                f"{len(uni)} blocks, between "
                f"{fmt(min(r['ratio_to_other_values'] for r in uni), 3)} and "
                f"{fmt(max(r['ratio_to_other_values'] for r in uni), 3)} times smaller than the "
                f"median over the other four values."
            )
        mc = minima.get("mean_control", [])
        if mc and levels.get(LINES[0]):
            parts.append(
                f"At the mean-control anchor the minimum moves to the {_agree(mc, 'value_cp10k')} "
                f"CP10K query in all {len(mc)} blocks, and the perturbed genes' median pooled "
                f"control level is "
                + " and ".join(
                    f"{fmt(levels[cl]['median_control_cp10k'])} CP10K in {LABEL[cl]}"
                    for cl in LINES if levels.get(cl)
                )
                + "."
            )
        ic = minima.get("individual_cells", [])
        if ic and levels.get(LINES[0], {}).get("cells_zero_fraction") is not None:
            by_value: dict[float, list[str]] = {}
            for r in ic:
                by_value.setdefault(r["value_cp10k"], []).append(MODEL_LABEL[r["model"]])
            where = "; ".join(
                f"{fmt(v)} CP10K for {', '.join(sorted(set(names)))}"
                for v, names in sorted(by_value.items())
            )
            parts.append(
                f"At the individual-cells anchor it sits at {where}. Of the perturbed gene and "
                "cell pairs, "
                + " and ".join(
                    f"{fmt(100 * levels[cl]['cells_zero_fraction'], 3)}% in {LABEL[cl]}"
                    for cl in LINES if levels.get(cl)
                )
                + " are at exactly zero counts, so the 0 CP10K query is already a no-op for a "
                "sizeable share of them."
            )
        s.blocks.append(
            Para(
                "Two forward passes that differ only in the perturbed gene's input value are "
                "identical when that value equals the gene's own level in the anchor, so the "
                "response has to be smallest at the grid point nearest that level. It is, which "
                "is the cheapest available check that a query means what it says. " + " ".join(parts)
            )
        )
    no_op_rows = []
    for (model, cl), b in sorted(bundles.items()):
        for anchor, queries in b.get("query_summary", {}).items():
            for r in queries:
                n_zero = r.get("n_no_op_queries")
                if n_zero:
                    no_op_rows.append(
                        [
                            MODEL_LABEL[model],
                            LABEL[cl],
                            ANCHOR_LABEL.get(anchor, anchor),
                            fmt(r["value_cp10k"]),
                            f"{fmt(n_zero)} of {fmt(r['n_queries'])}",
                            rate(n_zero / r["n_queries"]),
                        ]
                    )
    stored_zero = [
        blk
        for b in bundles.values()
        for blk in b.get("blocks", {}).values()
        if blk.get("all_zero_fraction")
    ]
    if no_op_rows:
        s.blocks += [
            Para(
                "A related quantity run 3 needs is how many queries the model could not see at "
                "all. When the value asked for is the value the input already carried, the two "
                "passes are the same call and the response is identically zero rather than "
                "small. It happens almost only at the 0 CP10K query, at the individual-cells "
                "anchor, where a gene that already reads zero in that cell is unchanged by "
                "being set to zero:"
            ),
            Table(
                ["model", "cell line", "anchor", "value (CP10K)", "no-op queries", "fraction"],
                no_op_rows,
            ),
        ]
        # each model's no-op count is read against its OWN perturbed set: with mixed
        # coverage the number of gene-and-cell pairs is not the same for every block
        ref_rows, matches_zero, matches_floor, matches_neither = [], [], [], []
        for (model, cl), b in sorted(bundles.items()):
            lv = levels_by_tag.get((tags[(model, cl)], cl))
            if not lv or lv.get("cells_zero_n") is None:
                continue
            observed = next(
                (
                    r.get("n_no_op_queries", 0)
                    for r in b.get("query_summary", {}).get("individual_cells", [])
                    if r["value_cp10k"] == 0.0
                ),
                None,
            )
            if observed is None:
                continue
            verdict = (
                "exactly-zero pairs"
                if observed == lv["cells_zero_n"]
                else "below-floor pairs"
                if observed == lv["cells_below_floor_n"]
                else "fewer than either"
            )
            (
                matches_zero if verdict.startswith("exactly")
                else matches_floor if verdict.startswith("below")
                else matches_neither
            ).append(MODEL_LABEL[model])
            ref_rows.append(
                [
                    MODEL_LABEL[model],
                    LABEL[cl],
                    fmt(lv["cells_n_pairs"]),
                    fmt(observed),
                    fmt(lv["cells_zero_n"]),
                    fmt(lv["cells_below_floor_n"]),
                    verdict,
                ]
            )
        if ref_rows:
            s.blocks += [
                Para(
                    "The count is set by how each model encodes its input rather than by the "
                    "cell, and it lands on one of two interpretable numbers: how many perturbed "
                    "gene and cell pairs read exactly zero counts, and how many sit below the "
                    f"{fmt(C.RUN1_FLOOR_CP10K)} CP10K read-out floor. Both are counted over each "
                    "block's own perturbed genes, which is why the number of pairs differs "
                    "between blocks at different coverage."
                ),
                Table(
                    ["model", "cell line", "gene and cell pairs", "no-ops at 0 CP10K",
                     "exactly zero counts", "below the read-out floor", "matches"],
                    ref_rows,
                ),
                Para(
                    (
                        f"{_join(sorted(set(matches_zero)))} matches the exactly-zero count, "
                        "which is what a raw-count input predicts: only a gene already at zero is "
                        "unchanged by being set to zero. "
                        if matches_zero
                        else ""
                    )
                    + (
                        f"{_join(sorted(set(matches_floor)))} matches the below-floor count, "
                        "which is what a binned input predicts: every below-floor pair falls in "
                        "its lowest bin. "
                        if matches_floor
                        else ""
                    )
                    + (
                        f"{_join(sorted(set(matches_neither)))} records fewer than either, so for "
                        "those even an already-zero gene is not always delivered the same way as "
                        "an explicit zero. "
                        if matches_neither
                        else ""
                    )
                    + (
                        "The single TranscriptFormer no-op at the 64 CP10K query is recorded and "
                        "not explained: it is not that model's "
                        f"{fmt(levels[LINES[0]]['cells_clip_cp10k'])} CP10K count clip, which "
                        f"{fmt(levels[LINES[0]]['cells_above_clip_n'])} {LABEL[LINES[0]]} pairs "
                        "sit at or above."
                        if (levels.get(LINES[0]) or {}).get("cells_above_clip_n")
                        else ""
                    )
                ),
            ]
        spread_rows = [
            [
                MODEL_LABEL[model],
                LABEL[cl],
                fmt(sp["n_cells"]),
                fmt(sp["median_cv_of_response_size"], 3),
                fmt(sp["p90_cv_of_response_size"], 3),
            ]
            for (model, cl), b in sorted(bundles.items())
            for anchor, sp in b.get("anchor_cell_spread", {}).items()
            if sp.get("median_cv_of_response_size") is not None
        ]
        if spread_rows:
            cvs = [float(r[3]) for r in spread_rows]
            s.blocks += [
                Para(
                    "The other thing that anchor hides is how much the choice of cell matters. "
                    "The stored block is the average over the four cells, so the spread across "
                    "them is recorded separately: per perturbed gene, the coefficient of "
                    "variation of the response size over the four cells."
                ),
                Table(
                    ["model", "cell line", "cells", "median CV over cells", "90th percentile"],
                    spread_rows,
                ),
                Para(
                    f"The median gene's response size moves by {rate(min(cvs))} to "
                    f"{rate(max(cvs))} of itself between cells, and the tail is wider still. "
                    "Four cells is enough to average out a no-op but not enough to make the "
                    "anchor a stable quantity, so a run 3 result that rests on the "
                    "individual-cells anchor should be read as a four-cell sample rather than "
                    "as the model's response in a real cell."
                ),
            ]
        s.blocks.append(
            Para(
                "Nothing in the stored features exposes this. "
                + (
                    "No stored response vector is identically zero, because the stored "
                    "individual-cells block averages over the four cells and a no-op in one "
                    "cell is diluted by the other three. "
                    if not stored_zero
                    else ""
                )
                + "A consumer that averages over these pairs is averaging a structural zero in "
                "with real effects, so run 3 should read the per-query record and drop them "
                "rather than treat a zero as a small response."
            )
        )
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
    ]
    mask_findings, shrink, grow = [], [], []
    for model in MODELS:
        pieces = []
        for cl in LINES:
            b = bundles.get((model, cl))
            if not b:
                continue
            un = b["blocks"].get("mean_control/zero_out", {}).get("abs_median")
            ma = b["blocks"].get("mean_control_masked/zero_out", {}).get("abs_median")
            if un and ma:
                pieces.append(
                    f"{fmt(un, 3)} to {fmt(ma, 3)} in {LABEL[cl]} (a factor of {rate(ma / un)})"
                )
                (shrink if ma < un else grow).append((MODEL_LABEL[model], ma / un))
        if pieces:
            mask_findings.append(f"{MODEL_LABEL[model]}: " + "; ".join(pieces))
    if mask_findings:
        names = lambda rows: ", ".join(sorted({n for n, _ in rows}))  # noqa: E731
        s.blocks += [
            Para(
                "Masking the read-out gene's own input value changes the response at the same "
                "anchor, and by enough that the two are different features rather than the same "
                "one with noise. Median absolute zero-out response, unmasked against masked: "
                + ". ".join(mask_findings)
                + ". The size of the gap differs by model rather than being a fixed offset, "
                "which is what the masked block was extracted to establish."
            ),
            Para(
                "The direction is not the same for all of them, which is the part worth stating "
                "plainly. "
                + (
                    f"For {names(grow)} the response grows when the gene's own value is taken "
                    f"away, up to a factor of {rate(max(r for _, r in grow))}, which is the "
                    "naive expectation: a model that cannot see the answer has to infer it from "
                    "the rest of the cell, so the rest of the cell matters more. "
                    if grow
                    else ""
                )
                + (
                    f"For {names(shrink)} it shrinks instead, by as much as a factor of "
                    f"{rate(min(r for _, r in shrink))}: removing the gene's own value made "
                    "those models less sensitive to the perturbed gene, not more. "
                    if shrink
                    else ""
                )
                + "Two accounts are consistent with the shrinking direction: a masked prediction "
                "falling back on a population prior that no single other gene moves much, and "
                "the mask token itself being off the distribution the model was trained to read. "
                "The next section separates them. A consumer should treat masked and unmasked as "
                "two features and not assume a common sign of the effect across models."
            ),
        ]
    if hidden:
        s.blocks += hidden_blocks(hidden)
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
    stable, unstable = [], []
    for (model, cl), b in sorted(bundles.items()):
        for anchor, sel in b["manifest"].get("step_selection", {}).items():
            (stable if sel["verdict"] != "magnitude_unstable" else unstable).append(
                (model, cl, anchor, sel)
            )
    if unstable:
        worst_model = sorted({m for m, _, _, _ in unstable})
        example = next(
            (s_ for m, cl, a, s_ in unstable if a == "mean_control" and cl == LINES[0]),
            unstable[0][3],
        )
        curve = example["adjacent_agreement"]
        smallest_abs = float(
            np.median(
                [
                    r["median_abs_value"]
                    for _, _, _, sel in unstable
                    for r in sel["adjacent_agreement"]
                ]
            )
        )
        s.blocks.append(
            Para(
                f"{len(stable)} of the {len(stable) + len(unstable)} block and anchor pairs "
                f"converge and {len(unstable)} do not, and the split is by model rather than by "
                f"anchor: every failure is "
                f"{', '.join(MODEL_LABEL[m] for m in worst_model)}. Its adjacent-step agreement "
                f"climbs steadily without ever reaching the threshold, from "
                f"{fmt(curve[0]['median_pearson_adjacent'])} at step {fmt(curve[0]['step'])} to "
                f"{fmt(curve[-1]['median_pearson_adjacent'])} at step {fmt(curve[-1]['step'])}, "
                f"so the rule falls back on the step with the best direction agreement "
                f"({fmt(example['selected_step'])}) and flags the block. The reason is visible in "
                f"the size of the numbers: across its swept steps the gradient columns being "
                f"differenced have a median absolute value of {fmt(smallest_abs, 3)}, close "
                f"enough to float noise that the "
                "difference between two nearby steps is dominated by it. Those gradient features "
                "are usable for direction and ranking and not for magnitude, and a consumer reads "
                "that verdict out of `step_selection.json` rather than inferring it."
            )
        )
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
                    [
                        MODEL_LABEL[model],
                        LABEL[cl],
                        ANCHOR_LABEL.get(anchor, anchor),
                        "production",
                        fmt(chk["median_pearson_two_sided"], 6),
                        fmt(chk["median_relative_l2_two_sided"]),
                        fmt(chk["step"]),
                    ]
                )
        untiled = b["manifest"].get("untiled_exact_check") or {}
        if untiled.get("ran"):
            exact_rows.append(
                [
                    MODEL_LABEL[model],
                    LABEL[cl],
                    "uniform virtual",
                    f"side check, {untiled['n_genes']} genes",
                    fmt(untiled["median_pearson_two_sided"], 6),
                    fmt(untiled["median_relative_l2_two_sided"]),
                    fmt(untiled["step"]),
                ]
            )
    if exact_rows:
        s.blocks += [
            Para(
                "Where an exact derivative exists it is the feature, and the central difference "
                "at the selected step is its cross-check. scPRINT admits forward-mode autodiff in "
                "the layout it is actually run in, so the cross-check runs on the production "
                "features. scGPT does not, because its read-out set is delivered in several "
                "tiled passes; the exact derivative exists only in the untiled layout, so that "
                "comparison is a side check on a smaller gene sample rather than a statement "
                "about the production features. The medians below are taken over the base values "
                "whose central difference stays inside the input range: at a base of 0 CP10K the "
                "difference would have to cross the floor, so that value is one-sided and "
                "excluded."
            ),
            Table(
                ["model", "cell line", "anchor", "layout", "median Pearson",
                 "median relative L2", "step"],
                exact_rows,
            ),
        ]
        s.blocks += aido_gradient_blocks(bundles, probes)
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
            "is emitted beside the raw one wherever the output has a bridge to a linear "
            "expression scale, which is every model except scGPT; and every block also carries a "
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
    ]
    coupling = []
    for model in MODELS:
        ratios = []
        for cl in LINES:
            b = bundles.get((model, cl))
            if not b:
                continue
            for queries in b.get("query_summary", {}).values():
                for r in queries:
                    if r["cross_gene_sd"]:
                        ratios.append(abs(r["cross_gene_mean"]) / r["cross_gene_sd"])
        if ratios:
            coupling.append((model, float(np.median(ratios)), max(ratios)))
    if coupling:
        s.blocks.append(
            Para(
                "Ratio of the absolute cross-gene mean response to its cross-gene spread, over "
                "every query in both cell lines: "
                + "; ".join(
                    f"{MODEL_LABEL[m]} median {fmt(med, 3)}, worst {fmt(hi, 3)}"
                    for m, med, hi in coupling
                )
                + ". The check is applied only to the two models whose head is a softmax over "
                "the supplied genes, "
                + _join([MODEL_LABEL[m] for m in vf.SOFTMAX_MODELS])
                + ". The reading is one-directional: a small ratio is what "
                "redistribution produces, but it is not by itself evidence of a softmax. "
                f"{MODEL_LABEL['aido_cell']}, whose head is a per-gene regression over the full "
                "transcriptome, sits in the same range simply because one gene's query moves a "
                "19,264-gene prediction very little on average. scGPT, whose output is a "
                "per-cell bin index, is the only one clearly above."
            )
        )
    tf_fail = [
        (cl, r["check"], r["detail"])
        for cl in LINES
        for r in (bundles.get(("transcriptformer", cl)) or {}).get("checks", [])
        if not r["ok"] and r["check"].endswith("softmax_coupling")
    ]
    if tf_fail:
        values = re.findall(r"= ([0-9.]+) \(tolerance", " ".join(d for _, _, d in tf_fail))
        tf_total = sum(
            1
            for cl in LINES
            for r in (bundles.get(("transcriptformer", cl)) or {}).get("checks", [])
            if r["check"].endswith("softmax_coupling")
        )
        s.blocks.append(
            Para(
                f"TranscriptFormer misses the {vf.SOFTMAX_MEAN_TOLERANCE} tolerance on "
                f"{len(tf_fail)} of its {tf_total} anchor and cell-line pairs, at "
                f"{', '.join(values)}. The cause is the layout rather than the head. Its "
                "read-out set does not fit in one pass, so it is delivered in two, each "
                "normalized by that pass's own total; the shares sum to one inside a pass and "
                "not across the union of the two, which leaves exactly this kind of small "
                "residual offset. Every block ships a recentred variant beside the raw one, and "
                "the recentred variant removes it. A consumer that cares about the identity "
                "should read `recentred`."
            )
        )
    s.blocks.append(FigRef("feature_scale"))
    scale = []
    for model in MODELS:
        vals = [
            b["blocks"]["mean_control/zero_out"]["abs_median"]
            for cl in LINES
            for b in [bundles.get((model, cl))]
            if b and "mean_control/zero_out" in b["blocks"]
        ]
        if vals:
            scale.append((model, min(vals), max(vals)))
    if len(scale) > 1:
        spread = max(hi for _, _, hi in scale) / min(lo for _, lo, _ in scale)
        s.blocks.append(
            Para(
                "The four raw scales are not comparable, and the gap is not small. Median "
                "absolute zero-out response at the mean-control anchor: "
                + "; ".join(
                    f"{MODEL_LABEL[m]} {fmt(lo, 3)} to {fmt(hi, 3)}" for m, lo, hi in scale
                )
                + f", about {int(round(spread)):,}-fold between the extremes, which reflects each "
                "head's units rather than anything biological. The comparison run should read "
                "the standardized form by default and treat a raw cross-model magnitude "
                "comparison as meaningless."
            )
        )
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
                "The three widths do not agree closely, so the production sweep uses the "
                f"{cc['production_mode']} width, the one inside scGPT's training length. The "
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
        Head(2, "Read-out order alignment"),
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
                    f"{rate(r['own_largest'])} ({rate(r['own_largest_null'])})",
                    f"{rank(r['self_rank_median'])} of {fmt(b['n_readout'])} "
                    f"({rank(r['self_rank_median_null'])})",
                    f"{rate(r['sign_rate'])} ({rate(r['sign_rate_null'])})",
                    ", ".join(r["separated_by"]) or "none",
                ]
            )
    guard_met, guard_missed = [], []
    for model in MODELS:
        rows_m = [
            r
            for cl in LINES
            for r in (bundles.get((model, cl)) or {}).get("alignment", [])
        ]
        if not rows_m:
            continue
        (guard_met if all(r["own_largest"] >= C.SELF_RESPONSE_MIN_PASS for r in rows_m)
         else guard_missed).append(MODEL_LABEL[model])
    if wire_rows:
        separated = [r for r in wire_rows if r[-1] != "none"]
        tally = (
            f"All {len(wire_rows)} block and anchor pairs separate from their null on at least "
            "one statistic."
            if len(separated) == len(wire_rows)
            else f"{len(separated)} of {len(wire_rows)} block and anchor pairs separate from "
            f"their null; the rest do not establish alignment and are not usable."
        )
        s.blocks += [
            Table(
                ["model", "cell line", "anchor", "perturbed genes sampled",
                 "own entry largest (null)", "median self-rank (null)",
                 "own output moves as expected (null)", "separated on"],
                wire_rows,
            ),
            Para(
                "Each cell gives the observed statistic with its permuted null in parentheses. "
                "The last column names the statistics that separate widely enough to establish "
                f"alignment for that block. {tally}"
            ),
            Para(
                "The pre-registered guard is reported as it stands and is not retro-fitted to "
                "the result. "
                + (
                    f"It is met by {_join(guard_met)} at every anchor in both cell lines and "
                    f"it is not met by {_join(guard_missed)}, so for those models the "
                    "alignment conclusion rests on the permutation-controlled statistics rather "
                    "than on the threshold written before the data existed. "
                    if guard_met and guard_missed
                    else ""
                )
                + "What the guard would have caught the replacements also catch: a misalignment "
                "relabels the own entry at random, which is exactly the permutation, and it "
                "would drive all three statistics to their nulls at once."
            ),
        ]
        anti = [
            (MODEL_LABEL[model], LABEL[cl], r)
            for (model, cl), b in sorted(bundles.items())
            for r in b.get("alignment", [])
            if r["sign_rate"] < 0.5 and r["sign_rate_null"] > 0.5
        ]
        if anti:
            worst = min(anti, key=lambda t: t[2]["sign_rate"])
            names = sorted({n for n, _, _ in anti})
            s.blocks.append(
                Para(
                    "One reading in that table is a finding rather than a check. For "
                    f"{_join(names)} the sign statistic does not merely miss its threshold, it "
                    "inverts: at worst, the perturbed gene's own predicted output rises with its "
                    f"input value for only {pct(worst[2]['sign_rate'])} of sampled genes, against "
                    f"{pct(worst[2]['sign_rate_null'])} when that same column is relabelled at "
                    "random. The inversion is specific to the "
                    "gene's own entry, which is why it is evidence of correct alignment and not "
                    "of a mix-up: other genes move the ordinary way. It is the behaviour of a "
                    "head trained to reconstruct expression from context, which pulls an "
                    "artificially raised gene back toward the level the rest of the cell "
                    "implies. Run 3 should not assume the own-entry sign is positive, and any "
                    "summary that averages the own entry together with the rest of the vector "
                    "will partly cancel."
                )
            )
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
            "  hidden_readout/condition=<c>/     visible / mask token / median level, where\n"
            "                                    the model has a masking regime\n"
            "\n"
            "features_full_scope/<model>/<cell_line>/   the same layout at full coverage\n"
            "feature_index/blocks.json                  the canonical block per (model, line)\n"
        ),
        Para(
            "Two trees, because coverage differs by model: `features/` holds the 250-gene blocks "
            "the first wave produced and `features_full_scope/` the full-coverage blocks that "
            "supersede them for the three cheap models. Read `feature_index/blocks.json` to "
            "resolve the canonical block for a (model, cell line) rather than globbing either "
            "tree, and read the `coverage` block of a manifest before comparing two blocks: they "
            "share the smaller one's perturbed genes but not its row order, so join on "
            "`perturbed_gene_id`."
        ),
        Para(
            "Each npz carries `raw` and `standardized` always, and `recentred` and `log2fc` "
            "wherever the model's output has a bridge to a linear expression scale, plus the "
            "perturbed and read-out gene identifiers and the value grid, so a block is "
            "self-describing."
        ),
        Head(3, "Checkpoints, and where model code had to be changed"),
        Table(
            ["model", "checkpoint", "revision"],
            [
                [
                    MODEL_LABEL[model],
                    f"`{b['manifest']['contract']['checkpoint']}`",
                    f"`{b['manifest']['contract']['checkpoint_revision']}`",
                ]
                for model in MODELS
                for b in [next((bundles[(model, cl)] for cl in LINES
                                if (model, cl) in bundles), None)]
                if b
            ],
        ),
    ]
    subs = [
        (MODEL_LABEL[model], sub)
        for model in MODELS
        for b in [next((bundles[(model, cl)] for cl in LINES if (model, cl) in bundles), None)]
        if b
        for sub in b["manifest"].get("substitutions", [])
    ]
    s.blocks.append(
        Para(
            "Both cell lines of a model load the same checkpoint at the same revision, and each "
            "block's `manifest.json` records it. "
            + (
                f"Every model needed some change to its shipped code to run here, {len(subs)} "
                "in total, and each block records its own verbatim. They are worth reading "
                "because several of them touch numerics:"
                if len({n for n, _ in subs}) == len([m for m in MODELS if any(
                    (m, cl) in bundles for cl in LINES)])
                else f"{_join(sorted({n for n, _ in subs}))} needed a change to the shipped "
                f"model code, {len(subs)} in total:"
            )
        )
    )
    if subs:
        s.blocks += [
            Table(["model", "substitution"], [[n, _cell(t)] for n, t in subs]),
            Para(
                "They fall into three kinds. Three are import plumbing, working around packages "
                "that will not install or configure here, and leave the model itself untouched. "
                "Three replace a fused half-precision attention kernel with the same attention "
                "in float32, because the fused kernel refuses float32 or carries no forward-mode "
                "derivative; the weights and the attention function are unchanged, the kernel and "
                "the dtype are not, so exact numerics differ from a flash-attention run. "
                "AIDO.Cell's is in that second group and also repairs a bug: "
                "its packaged non-flash fallback softmaxes across heads rather than across the "
                "sequence, which is not the attention the weights were trained with. No "
                "cross-check against the published kernel was possible for it, because that "
                "kernel does not import in this environment. The last one remaps scGPT's fused "
                "query-key-value tensors onto torch's packed projection, which is the only change "
                "that moves checkpoint weights, and it was verified numerically at load time."
            ),
        ]
    if verification:
        checks = verification["checks"]
        ok = sum(1 for r in checks if r["ok"])
        block_total = sum(len(b["checks"]) for b in bundles.values())
        block_ok = sum(1 for b in bundles.values() for r in b["checks"] if r["ok"])
        s.blocks += [
            Head(2, "Verification"),
            Para(
                "Verification runs at two levels. Every block is checked inside a job that "
                "mounts the volume the extraction wrote, against that block's own npz files "
                f"rather than against a summary: {block_total} checks over "
                f"{len(bundles)} blocks, {block_ok} passed. The roll-up then reads the "
                f"{len(bundles)} per-block records together and adds coverage and alignment "
                f"checks: {len(checks)} checks, {ok} passed. Both outputs ship with the report, "
                "and the failures are listed here rather than summarised away."
            ),
        ]
        failing_kinds = {}
        for b in bundles.values():
            for r in b["checks"]:
                if not r["ok"]:
                    kind = r["check"].split("/")[-1]
                    failing_kinds.setdefault(kind, []).append(f"{b['model']}/{b['cell_line']}")
        explain = {
            "wiring_magnitude": (
                "the pre-registered 95% own-entry-largest guard. Reported as it stands; the "
                "permutation-controlled statistics in the wiring section establish alignment "
                "for these blocks instead."
            ),
            "softmax_coupling": (
                "the redistribution identity, missed by a small margin because the read-out set "
                "is delivered in two separately normalized passes. The recentred variant that "
                "ships in every block removes the offset."
            ),
        }
        if failing_kinds:
            s.blocks.append(
                Table(
                    ["failing check", "blocks", "what it means"],
                    [
                        [
                            kind,
                            f"{len(where)} anchor and block pairs in "
                            f"{', '.join(sorted(set(where)))}",
                            explain.get(kind, "see the per-block record"),
                        ]
                        for kind, where in sorted(failing_kinds.items())
                    ],
                )
            )
        failed = [r for r in checks if not r["ok"]]
        if failed:
            s.blocks += [
                Para(
                    "The roll-up's own failures, which are the per-block failures restated plus "
                    "anything missing from the tree:"
                ),
                Table(
                    ["check", "detail"],
                    [[f"{r['section']}/{r['check']}", r["detail"]] for r in failed],
                ),
            ]
    sections.append(s)
    return sections


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
STYLE = """
:root {
  --ink: #1D272A; --muted: #5A6A6E; --rule: #D9D5CE; --paper: #FBFAF8;
  --accent: #C4650D; --panel: #F3F1EC;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--paper); color: var(--ink);
  font-family: 'Suisse Intl', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
  font-size: 16px; line-height: 1.62; -webkit-font-smoothing: antialiased;
}
main { max-width: 52rem; margin: 0 auto; padding: 3.5rem 1.5rem 5rem; }
h1 { font-size: 2.1rem; line-height: 1.2; letter-spacing: -0.01em; margin: 0 0 1.5rem; }
h2 {
  font-size: 1.45rem; line-height: 1.25; letter-spacing: -0.005em;
  margin: 3.25rem 0 1rem; padding-top: 1.25rem; border-top: 1px solid var(--rule);
}
h3 { font-size: 1.05rem; margin: 2rem 0 .6rem; color: var(--muted);
     text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }
section.figure h3.figure-title {
  text-transform: none; letter-spacing: 0; color: var(--ink); font-size: 1.02rem;
  font-weight: 600; margin: 0 0 .75rem;
}
p { margin: 0 0 1.1rem; }
ul { margin: 0 0 1.2rem; padding-left: 1.15rem; }
li { margin-bottom: .4rem; }
strong { font-weight: 600; }
code {
  font-family: 'SF Mono', ui-monospace, Menlo, Consolas, monospace; font-size: .88em;
  background: var(--panel); padding: .1em .35em; border-radius: 3px;
}
pre {
  background: var(--panel); border: 1px solid var(--rule); border-radius: 4px;
  padding: 1rem 1.1rem; overflow-x: auto; font-size: .84rem; line-height: 1.5;
}
pre code { background: none; padding: 0; }
table {
  width: 100%; border-collapse: collapse; margin: 0 0 1.6rem;
  font-size: .87rem; font-variant-numeric: tabular-nums;
}
th, td { text-align: left; padding: .42rem .6rem; border-bottom: 1px solid var(--rule); }
th { font-weight: 600; color: var(--muted); border-bottom: 1.5px solid var(--ink);
     white-space: nowrap; }
tr:last-child td { border-bottom: 1px solid var(--ink); }
section.figure {
  margin: 2rem 0 2.5rem; padding: 1.25rem 1.25rem .5rem;
  background: #FFFFFF; border: 1px solid var(--rule); border-radius: 5px;
}
.figure-description, .figure-findings { font-size: .87rem; color: var(--muted); line-height: 1.55; }
.figure-findings { color: var(--ink); }
small { color: var(--muted); }
@media (max-width: 40rem) {
  main { padding: 2rem 1rem 3rem; }
  table { font-size: .8rem; display: block; overflow-x: auto; }
}
"""


def render_html(sections: list[Section], figs: dict) -> str:
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_esc(TITLE)}</title>",
        f"<style>{STYLE}</style>",
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

    # a small machine-readable side output: which (model, cell line, anchor, value)
    # queries were no-ops, so run 3 can drop them instead of averaging structural zeros
    pd.DataFrame(
        [
            {
                "model": model,
                "cell_line": cl,
                "anchor": anchor,
                "value_cp10k": r["value_cp10k"],
                "n_queries": r["n_queries"],
                "n_no_op_queries": r.get("n_no_op_queries", 0),
            }
            for (model, cl), b in sorted(bundles.items())
            for anchor, queries in b.get("query_summary", {}).items()
            for r in queries
        ]
    ).to_csv(out / "no_op_queries.csv", index=False)

    hidden = load_hidden(repo)
    probes = load_cost_probes(repo)
    print(
        f"[report] {len(hidden)} hidden-readout records, {len(probes)} cost probes", flush=True
    )
    figs = figmod.build_all(bundles, repo)
    sections = build_sections(bundles, floor, verification, hidden=hidden, probes=probes)
    (out / "feature_extraction_report.html").write_text(render_html(sections, figs))
    (out / "feature_extraction_report.md").write_text(render_markdown(sections, figs))
    print(f"[report] wrote {out / 'feature_extraction_report.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
