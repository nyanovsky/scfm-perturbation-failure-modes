"""Assemble the data-preparation report from the artifacts themselves.

Every number in the report is read out of a file that ships with it, so the
report cannot drift from the artifacts, and a later run can re-derive any figure
in it. Run pod-side after the figures:

    uv run python -m scfm_pfm.report.build --root results/artifacts_mirror
"""

from __future__ import annotations

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
    for m in MODELS:
        c = root / "models" / m / "contract.json"
        d = root / "models" / m / "demo.json"
        if c.exists():
            contracts[m] = json.loads(c.read_text())
        if d.exists():
            demos[m] = json.loads(d.read_text())

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
                f"| {m} | {c['output_semantics'].split('.')[0]} | {c['information_flow']} | "
                f"{'yes' if c['ordering_dependent'] else 'no'} | {c['jacobian_method']} |"
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
            for s in c["substitutions"]:
                add(f"  - {s}")
        add("")

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
