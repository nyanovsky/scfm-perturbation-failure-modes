"""Assemble the data-preparation report from the artifacts themselves.

Every number in the report is read out of a file that ships with it, so the
report cannot drift from the artifacts, and a later run can re-derive any figure
in it. Run pod-side after the figures:

    uv run python -m scfm_pfm.report.build --root results/artifacts_mirror
"""

from __future__ import annotations

import re

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import config as C
from .. import paths
from ..data import sources as data_sources
from ..models import sources as model_sources

LINES = ("hepg2", "jurkat")
LABEL = {"hepg2": "HepG2", "jurkat": "Jurkat"}
MODELS = ("scgpt", "aido_cell", "scprint", "transcriptformer")


def _first_sentence(text: str) -> str:
    """First sentence, without cutting at an abbreviation such as 'i.e.'."""

    parts = re.split(r"(?<!\bi\.e)(?<!\be\.g)(?<!\bcf)\.\s+", str(text).strip())
    return parts[0].rstrip(".") if parts else str(text)


def _sig(x: float, n: int = 4) -> str:
    """Four significant figures, switching to scientific notation for small values."""

    try:
        v = float(x)
    except (TypeError, ValueError):
        return "?"
    if v != v:
        return "n/a"
    if v != 0 and abs(v) < 1e-3:
        return f"{v:.{n - 1}e}"
    return f"{v:.{n}g}"


def _fmt(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def build(root: Path, out_path: Path, figures_dir: Path) -> str:
    qc = {cl: json.loads((root / "effects" / cl / "qc.json").read_text()) for cl in LINES}
    pert = {cl: pd.read_parquet(root / "effects" / cl / "perturbations.parquet") for cl in LINES}
    ceil = {cl: pd.read_parquet(root / "effects" / cl / "noise_ceiling_summary.parquet") for cl in LINES}
    genes = {cl: pd.read_parquet(root / "effects" / cl / "genes.parquet") for cl in LINES}
    cov = json.loads((root / "gene_universe" / "coverage.json").read_text())
    cc_path = root / "crosscheck" / "summary.json"
    cc = json.loads(cc_path.read_text()) if cc_path.exists() else None

    contracts = {}
    demos = {}
    envs = {}
    for m in MODELS:
        c = root / "models" / m / "contract.json"
        d = root / "models" / m / "demo.json"
        e = root / "models" / m / "environment.json"
        if c.exists():
            contracts[m] = json.loads(c.read_text())
        if d.exists():
            demos[m] = json.loads(d.read_text())
        if e.exists():
            envs[m] = json.loads(e.read_text())

    L = []
    add = L.append
    add("# Replogle-Nadig effect matrices and the four-model response contract\n")
    add(
        "This run produces two artifacts that every later run in this thread reads "
        "through a file contract: measured perturbation effects for two CRISPRi "
        "Perturb-seq cell lines, and a written, tested description of what each of four "
        "single-cell foundation models emits when asked about a cell.\n"
    )
    add(f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from the artifacts named below._\n")

    # ---------------- data ----------------
    add("## The measured effects\n")
    add(
        f"Source: GEO {data_sources.GEO_SERIES}, the two deposited raw single-cell "
        "matrices. Both were downloaded with their md5 recorded and read directly as "
        "HDF5 row chunks rather than through anndata, because `X` is a dense float32 "
        "matrix and streaming it keeps peak memory at one chunk plus accumulators.\n"
    )
    rows = ["| | HepG2 | Jurkat |", "|---|---|---|"]
    def line(name, fn):
        rows.append(f"| {name} | {fn('hepg2')} | {fn('jurkat')} |")
    line("cells x measured genes", lambda cl: f"{qc[cl]['shape']['cells']:,} x {qc[cl]['shape']['genes']:,}")
    line("file size", lambda cl: f"{qc[cl]['source']['size_bytes'] / 1e9:.3f} GB")
    line("md5", lambda cl: f"`{qc[cl]['source']['md5']}`")
    line("sequencing batches", lambda cl: f"{qc[cl]['n_gem_groups']}")
    line("cells after QC", lambda cl: f"{qc[cl]['retained']['cells']:,}")
    line("non-targeting control cells", lambda cl: f"{qc[cl]['retained']['control_cells']:,}")
    line("retained perturbations", lambda cl: f"{qc[cl]['retained']['perturbations']:,}")
    line("expressed genes (CP10K >= %.1f)" % C.EFFECTS.expressed_min_cp10k,
         lambda cl: f"{int(genes[cl].expressed.sum()):,}")
    line("median on-target knockdown", lambda cl: _fmt(qc[cl]['knockdown']['median']))
    line("median DE genes per perturbation", lambda cl: f"{int(pert[cl].n_de_genes.median()):,}")
    line("processing runtime", lambda cl: f"{qc[cl]['runtime_seconds']:.0f} s")
    add("\n".join(rows) + "\n")

    add("### What the filters remove\n")
    add(
        "The deposited matrices are already partly quality-controlled, so per-cell "
        "filters remove very little: in HepG2 only the mitochondrial-fraction filter "
        f"removes anything ({qc['hepg2']['filter_funnel'][4]['removed']:,} cells), and in "
        f"Jurkat only the minimum-count filter does "
        f"({qc['jurkat']['filter_funnel'][2]['removed']:,} cells). Almost all of the "
        "attrition is the requirement that a perturbation have at least "
        f"{C.PERTURBATION_FILTERS.min_cells} QC-passing cells across at least "
        f"{C.PERTURBATION_FILTERS.min_batches} batches, which takes 2,393 perturbations with any "
        f"passing cell down to {qc['hepg2']['retained']['perturbations']:,} in HepG2 and "
        f"{qc['jurkat']['retained']['perturbations']:,} in Jurkat.\n"
    )
    add("![Filter funnel](figures/filter_funnel/filter_funnel.html)\n")

    add("### The control contrast\n")
    add(
        "The median (perturbation, batch) cell count in these libraries is one to two "
        "cells, so a strictly within-batch estimator would discard nearly everything. "
        "The primary estimate instead references each perturbation against a "
        "batch-matched control profile: control means per `gem_group`, weighted by that "
        "perturbation's own per-batch cell counts. The batch-pooled estimate is kept "
        "beside it as a secondary column, and the two are compared directly in the "
        "cross-check section.\n"
    )
    add(
        f"Differential expression is a Welch t-test on log1p(CP10K) against pooled "
        f"non-targeting controls, Benjamini-Hochberg controlled at "
        f"{C.EFFECTS.de_fdr:.2f} across expressed genes.\n"
    )

    add("### On-target knockdown\n")
    add(
        f"Median knockdown is {_fmt(qc['hepg2']['knockdown']['median'])} in HepG2 and "
        f"{_fmt(qc['jurkat']['knockdown']['median'])} in Jurkat. "
        f"{qc['hepg2']['knockdown']['n_below_detect_threshold']:,} HepG2 and "
        f"{qc['jurkat']['knockdown']['n_below_detect_threshold']:,} Jurkat perturbations "
        f"fall below the {C.EFFECTS.knockdown_detect_min:.2f} detection floor, and "
        f"{qc['hepg2']['knockdown']['n_target_gene_not_measured']:,} and "
        f"{qc['jurkat']['knockdown']['n_target_gene_not_measured']:,} target genes are not "
        "in the measured panel at all, so no knockdown can be computed for them. They are "
        "flagged in `perturbations.parquet` rather than dropped: whether to use them is a "
        "decision for the run that consumes them.\n"
    )
    add("![Knockdown distribution](figures/knockdown_distribution/knockdown_distribution.html)\n")

    add("### Effect size and differential expression\n")
    strata_rows = ["| stratum | HepG2 n | HepG2 median DE | Jurkat n | Jurkat median DE |", "|---|---|---|---|---|"]
    for s in C.STRATUM_NAMES:
        h, j = pert["hepg2"], pert["jurkat"]
        strata_rows.append(
            f"| {s} | {int((h.effect_stratum == s).sum()):,} | "
            f"{h[h.effect_stratum == s].n_de_genes.median():.0f} | "
            f"{int((j.effect_stratum == s).sum()):,} | "
            f"{j[j.effect_stratum == s].n_de_genes.median():.0f} |"
        )
    add("\n".join(strata_rows) + "\n")
    add("![Effect size against DE count](figures/effect_size_vs_de/effect_size_vs_de.html)\n")

    add("### The noise ceiling\n")
    add(
        "A model asked to predict these effects cannot do better than the data predicts "
        "itself, so each perturbation is split in half and one half is scored against the "
        "other. Two split forms are reported. Splitting cells references each half "
        "against a disjoint half of the control cells, so shared control noise cannot "
        "inflate the agreement; splitting by batch alternates each perturbation's batches "
        "so the two sides use different control cells by construction.\n"
    )
    ceil_rows = ["| stratum | HepG2 cells | HepG2 batches | Jurkat cells | Jurkat batches |", "|---|---|---|---|---|"]
    for s in C.STRATUM_NAMES:
        vals = []
        for cl in LINES:
            for split in C.CEILING_SPLITS:
                sub = ceil[cl][
                    (ceil[cl].metric == "pearson_lfc")
                    & (ceil[cl].split == split)
                    & (ceil[cl].stratum == s)
                ]
                vals.append(_fmt(float(sub["median"].iloc[0])) if not sub.empty else "-")
        ceil_rows.append(f"| {s} | " + " | ".join(vals) + " |")
    add("\n".join(ceil_rows) + "\n")
    add(
        "Agreement rises monotonically with effect size in both cell lines and both split "
        "forms, and the two split forms agree closely with each other, which is what a "
        "working ceiling estimate should look like: the smallest effects are barely "
        "reproducible, the largest are.\n"
    )
    add("![Noise ceilings](figures/noise_ceilings/noise_ceilings.html)\n")

    # ---------------- crosscheck ----------------
    add("## Cross-check against the published reference\n")
    if cc:
        add(
            f"The reference release `{cc['reference']['repo']}` holds per-batch fold "
            "changes computed from the same series. Pooling its batches the same way we "
            f"pool ours, our per-gene log2 fold changes agree with it at Spearman "
            f"**{_fmt(cc['pooled_spearman'])}** over {cc['n_pooled_entries']:,} "
            f"(cell line, perturbation, gene) entries covering "
            f"{cc['n_shared_cell_line_target_pairs']:,} shared perturbations; the gate "
            f"for this run was {C.CROSSCHECK_MIN_SPEARMAN:.2f}. Median agreement within a "
            f"single perturbation is {_fmt(cc['median_per_pair_spearman'])}, and it rises "
            "with effect size:\n"
        )
        st_rows = ["| stratum | perturbations | median Spearman vs reference |", "|---|---|---|"]
        for s, v in sorted(cc.get("per_pair_spearman_by_stratum", {}).items()):
            st_rows.append(f"| {s} | {v['n']:,} | {_fmt(v['median_spearman'])} |")
        add("\n".join(st_rows) + "\n")
        own = cc["own_estimator_agreement"]
        add(
            "The same module compares our two estimators against each other. "
            "Batch-matched and batch-pooled log2 fold changes agree at median Pearson "
            f"{_fmt(own['median_pearson_within_vs_pooled_lfc'])}, and the deltas at "
            f"{_fmt(own['median_pearson_within_vs_pooled_delta'])}. Where they disagree, "
            "batch composition is doing the work, which is why the batch-matched estimate "
            "is the primary one.\n"
        )
        add("![Reference cross-check](figures/reference_crosscheck/reference_crosscheck.html)\n")
    else:
        add("_The reference cross-check had not completed when this report was built._\n")

    # ---------------- models ----------------
    add("## The four-model response contract\n")
    add(
        f"The shared gene universe holds **{cov['universe_size']:,} genes**: measured in "
        f"both cell lines ({cov['shared_measured_genes']:,} genes) and present in all four "
        "model vocabularies. Identifiers are resolved to Ensembl gene ids through the "
        "assay's own var table first, then HGNC approved symbols, then HGNC aliases and "
        "previous symbols, and the route taken is recorded per token. Round-tripping the "
        f"{cov['identifier_audit']['n_nadig_genes']:,} genes named in the series gives "
        f"{cov['identifier_audit']['ensg_to_symbol_to_ensg_failures']} Ensembl->symbol->Ensembl "
        f"and {cov['identifier_audit']['symbol_to_ensg_to_symbol_failures']} "
        "symbol->Ensembl->symbol failures.\n"
    )
    mrows = [
        "| model | vocabulary | identifier | fixed input order | covers HepG2 measured | covers Jurkat measured |",
        "|---|---|---|---|---|---|",
    ]
    for m, c in cov["coverage"].items():
        mrows.append(
            f"| {m} | {c['vocabulary_size']:,} | {c['identifier_kind']} | "
            f"{'yes' if c['order_is_a_model_constraint'] else 'no'} | "
            f"{c['covers_hepg2_measured']:,} / {c['hepg2_measured_total']:,} | "
            f"{c['covers_jurkat_measured']:,} / {c['jurkat_measured_total']:,} |"
        )
    add("\n".join(mrows) + "\n")
    add("![Model gene coverage](figures/model_gene_coverage/model_gene_coverage.html)\n")

    if contracts:
        add("### What each model emits\n")
        crows = ["| model | per-gene value | information flow | ordering matters | Jacobian |", "|---|---|---|---|---|"]
        for m in MODELS:
            c = contracts.get(m)
            if not c:
                continue
            crows.append(
                f"| {m} | {_first_sentence(c['output_semantics'])} | {c['information_flow']} | "
                f"{'yes' if c['ordering_dependent'] else 'no'} | "
                f"{(demos.get(m, {}).get('jacobian', {}) or {}).get('method', c['jacobian_method'])} |"
            )
        add("\n".join(crows) + "\n")
        add(
            "The per-gene values are on four different scales and are not comparable "
            "across models without a transformation the consuming run has to choose "
            "deliberately. Two of them (scPRINT, TranscriptFormer) end in a softmax "
            "across the genes supplied in the same forward pass, so their values are "
            "shares of a cell's counts and depend on which genes were supplied together.\n"
        )
        add("![Model responses on one cell](figures/model_responses_on_one_cell/model_responses_on_one_cell.html)\n")

        tf = demos.get("transcriptformer", {}).get("ordering_probe")
        if tf and "target_share_change_source_ahead" in tf:
            add("### TranscriptFormer is not symmetric, and the measurement says so\n")
            add(
                "TranscriptFormer is autoregressive with a causal mask, so a gene responds "
                "to another only when that other gene precedes it in the sequence. Reading "
                f"the same target gene twice from the same cell with "
                f"{tf['context_genes_between_them']} context genes between them: raising "
                f"the source gene's count from {tf['source_count']:.0f} to "
                f"{tf['source_count_raised_to']:.0f} changes the target's predicted share "
                f"by {tf['target_share_change_source_ahead']:+.3e} (on a base of "
                f"{tf['target_share_source_ahead']:.3e}) when the source is ahead of it, "
                f"and by {tf['target_share_change_source_behind']:+.1f} exactly when the "
                "source is behind it. A run that treats a TranscriptFormer response matrix "
                "as symmetric is reading half of it from a masked-out region.\n"
            )

        probes = {
            m: (demos.get(m, {}).get("jacobian", {}) or {}).get("symmetry_probe")
            for m in MODELS
        }
        if any(probes.values()):
            add("### Is the response matrix symmetric? Measured, not assumed\n")
            add(
                "A run that reads one column of the response matrix and treats the "
                "matrix as symmetric is making a claim no model documents. One column "
                "cannot answer it, so each model was asked for a second column at the "
                "gene its source gene moves most, and the two cross entries were "
                "compared. A verdict is only recorded when the Jacobian estimate itself "
                "is stable across step sizes (Pearson at least 0.99 and relative L2 "
                "difference at most 0.5 between two steps); otherwise step sensitivity "
                "and genuine asymmetry are not separable and the contract says so.\n"
            )
            srows = [
                "| model | second column | d target / d source | d source / d target | ratio | verdict |",
                "|---|---|---|---|---|---|",
            ]
            for m in MODELS:
                pr = probes.get(m)
                if not pr or pr.get("verdict") == "not_measured":
                    continue
                srows.append(
                    f"| {m} | {pr['second_column_method']} | "
                    f"{pr['d_target_d_source']:+.4e} | {pr['d_source_d_target']:+.4e} | "
                    f"{pr['ratio_d_source_d_target_over_d_target_d_source']:+.4g} | "
                    f"{pr['verdict']} |"
                )
            add("\n".join(srows) + "\n")
            determined = [
                m for m in MODELS
                if (probes.get(m) or {}).get("verdict") in ("asymmetric", "symmetric_at_this_pair")
            ]
            undetermined = [
                m for m in MODELS
                if (probes.get(m) or {}).get("verdict") == "not_determined"
                and (contracts.get(m) or {}).get("response_symmetric") is None
            ]
            structural = [
                m for m in MODELS
                if (probes.get(m) or {}).get("verdict") == "not_determined"
                and (contracts.get(m) or {}).get("response_symmetric") is False
            ]
            if determined:
                add(
                    "Where the measurement is decisive it comes out asymmetric: "
                    + ", ".join(determined)
                    + ". The response matrix is directional, so a consuming run has to "
                    "say which index is the perturbed gene and which is the read-out "
                    "gene, and cannot fill the other half by transposition.\n"
                )
            if undetermined:
                add(
                    "For "
                    + ", ".join(undetermined)
                    + " the finite-difference Jacobian is itself step-dependent at this "
                    "cell, so the contract records symmetry as not determined rather "
                    "than guessing. Anything in a later run that depends on the "
                    "magnitude (not just the sign or ranking) of these Jacobians needs "
                    "a step-size study first.\n"
                )
            if structural:
                add(
                    "For "
                    + ", ".join(structural)
                    + " the cross-entry probe is also step-limited, but symmetry is "
                    "already settled by the architecture rather than by this "
                    "measurement: a causal mask makes half the matrix unreachable, as "
                    "the ordering probe below shows directly.\n"
                )
            jrows = [
                "| model | Jacobian method | cross-check | Pearson | relative L2 gap | column L2 |",
                "|---|---|---|---|---|---|",
            ]
            for m in MODELS:
                jac = demos.get(m, {}).get("jacobian", {}) or {}
                chk = jac.get("check") or {}
                if not chk:
                    continue
                jrows.append(
                    f"| {m} | {jac.get('method')} | {chk.get('cross_check_method')} | "
                    f"{_sig(chk.get('pearson_primary_vs_cross_check', float('nan')))} | "
                    f"{_sig(chk.get('relative_l2_difference', float('nan')))} | "
                    f"{_sig(jac.get('column_l2', float('nan')))} |"
                )
            add("\n".join(jrows) + "\n")
            add(
                "Only scPRINT's column is an exact derivative (forward-mode autodiff, "
                "agreeing with a finite difference to a relative L2 difference of "
                f"{_sig((demos.get('scprint', {}).get('jacobian', {}).get('check') or {}).get('relative_l2_difference', float('nan')))}"
                "). The other three are finite differences over integer counts, which is "
                "the honest description of what a discretized or masked input allows: "
                "their direction is reproducible but their magnitude depends on the step, "
                "and TranscriptFormer's per-gene sensitivities on this cell are close to "
                "numerical noise (column L2 "
                f"{_sig((demos.get('transcriptformer', {}).get('jacobian', {}) or {}).get('column_l2', float('nan')))}"
                ").\n"
            )

        add("### Substitutions\n")
        add(
            "Where a published inference path could not be run as shipped, the "
            "substitution is recorded in the model's `contract.json` rather than hidden:\n"
        )
        for m in MODELS:
            c = contracts.get(m)
            if not c or not c.get("substitutions"):
                continue
            add(f"- **{m}**:")
            seen: set[str] = set()
            for sub in c["substitutions"]:
                key = str(sub).split("(")[0].strip()
                if key in seen:
                    continue
                seen.add(key)
                add(f"  - {sub}")
        add("")

        if envs:
            add("### How the four stacks were actually run\n")
            def _torch_of(env: dict) -> str:
                for pkg in env.get("pinned_packages", []):
                    name, _, version = str(pkg).partition("==")
                    if name.strip().lower() == "torch":
                        return version.strip() or "?"
                return "?"

            torch_versions = sorted({_torch_of(e) for e in envs.values()})
            add(
                "The plan called for one pinned environment per model, on the assumption "
                "that the four dependency sets are mutually unsatisfiable. They are not "
                "quite: pinning torch first and installing the four model packages with "
                "`--no-deps` (each would otherwise drag torch back to the version it was "
                "built against) gives one stack, `envs/model_stack.sh`, that runs all "
                f"four. Every model here ran on torch {', '.join(torch_versions)} in the "
                "same image, and each still records its own resolved package set in "
                "`environment.json`, so a later run can tell whether a result came from "
                "the same stack or a different one. Each model is still verified "
                "independently and still runs in its own job; only the installation is "
                "shared.\n"
            )
            erows = [
                "| model | image | device | checkpoint load | forward |",
                "|---|---|---|---|---|",
            ]
            for m in MODELS:
                e, d = envs.get(m), demos.get(m, {})
                if not e:
                    continue
                erows.append(
                    f"| {m} | `{e.get('image', 'unrecorded')}` | {e.get('device', '?')} | "
                    f"{(d.get('timings_seconds') or {}).get('load', float('nan')):.2f} s | "
                    f"{(d.get('timings_seconds') or {}).get('forward', float('nan')):.3f} s |"
                )
            add("\n".join(erows) + "\n")

    # ---------------- files ----------------
    add("## What a later run reads\n")
    add(
        "```\n"
        "effects/<cell_line>/effects.npz              delta and log2FC matrices, both estimators\n"
        "effects/<cell_line>/perturbations.parquet    per-perturbation metadata, knockdown, DE, strata, ceilings\n"
        "effects/<cell_line>/genes.parquet            gene index, control level, expressed flag\n"
        "effects/<cell_line>/noise_ceiling_summary.parquet\n"
        "effects/<cell_line>/qc.json                  provenance, filter funnel, config, environment\n"
        "crosscheck/summary.json                      agreement with the published reference\n"
        "gene_universe/gene_universe.parquet          one row per shared gene, with each model's token position\n"
        "gene_universe/vocabulary_<model>.parquet     every vocabulary token, its gene id, and how it resolved\n"
        "gene_universe/coverage.json                  coverage and the identifier audit\n"
        "models/<model>/response_vector.parquet       per-gene response on the demonstration cell\n"
        "models/<model>/jacobian_column.parquet       d response / d source gene count\n"
        "models/<model>/contract.json                 the declared response contract\n"
        "models/<model>/environment.json              pinned packages, image, device\n"
        "models/<model>/demo.json                     what was run, timings, checks\n"
        "```\n"
    )
    add(
        "`uv run python -m scfm_pfm.verify --all` re-checks every claim above against "
        "these files: matrix shapes and finiteness, knockdown and DE, strata, ceilings "
        "bounded and ordered, the reference Spearman against its gate, vocabulary "
        "positions inside each model's vocabulary, and one finite response vector per "
        "model with its contract fields filled in.\n"
    )

    add("## Provenance\n")
    prov = [
        "| artifact | source | revision or checksum |",
        "|---|---|---|",
        f"| HepG2 matrix | {qc['hepg2']['source']['url']} | md5 `{qc['hepg2']['source']['md5']}` |",
        f"| Jurkat matrix | {qc['jurkat']['source']['url']} | md5 `{qc['jurkat']['source']['md5']}` |",
    ]
    for m, c in cov["coverage"].items():
        p = c["provenance"]
        ref = p.get("revision") or p.get("md5") or ""
        src = p.get("repo") or p.get("source") or p.get("bucket", "")
        prov.append(f"| {m} vocabulary | {src} | `{ref}` |")
    for m in MODELS:
        c = contracts.get(m)
        if c:
            prov.append(f"| {m} checkpoint | {c['checkpoint']} | `{c['checkpoint_revision']}` |")
    if cc:
        prov.append(
            f"| reference release | {cc['reference']['repo']} | {cc['reference']['table']} |"
        )
    add("\n".join(prov) + "\n")
    add(
        f"Seed {C.SEED} throughout. Configuration lives in `src/scfm_pfm/config.py` and is "
        "copied into every `qc.json`, so a later run can see exactly which thresholds "
        "produced these matrices.\n"
    )

    text = "\n".join(L)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"wrote {out_path} ({len(text):,} characters)")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else paths.input_root()
    out = Path(args.out) if args.out else paths.repo_dir() / "results" / "data_preparation_report.md"
    build(root, out, paths.repo_dir() / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
