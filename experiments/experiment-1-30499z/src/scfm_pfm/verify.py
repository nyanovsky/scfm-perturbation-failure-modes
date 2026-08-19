"""The done condition for this run: ``uv run python -m scfm_pfm.verify --all``.

Every check is a claim a later run is entitled to rely on, so each prints the
number it checked rather than just PASS. A check that cannot run (missing file)
fails rather than being skipped, except for the sections excluded by the flags.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import paths
from .data.ceilings_grid import ADAPTIVE_K_MAX, ADAPTIVE_K_MIN, K_GRID

MODELS = ("scgpt", "aido_cell", "scprint", "transcriptformer")


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []

    def add(self, section: str, name: str, ok: bool, detail: str) -> bool:
        self.rows.append((section, name, bool(ok), detail))
        return bool(ok)

    @property
    def failed(self) -> list[tuple[str, str, bool, str]]:
        return [r for r in self.rows if not r[2]]

    def report(self) -> str:
        w = max(len(f"{s}/{n}") for s, n, _, _ in self.rows) if self.rows else 10
        lines = []
        for section, name, ok, detail in self.rows:
            lines.append(f"  {'PASS' if ok else 'FAIL'}  {f'{section}/{name}':<{w}}  {detail}")
        return "\n".join(lines)


def check_effects(ch: Checks, root: Path) -> None:
    for cl in paths.CELL_LINES:
        d = root / "effects" / cl
        sec = f"effects:{cl}"
        need = ["effects.npz", "genes.parquet", "perturbations.parquet", "noise_ceiling_summary.parquet", "qc.json"]
        missing = [f for f in need if not (d / f).exists()]
        if not ch.add(sec, "files", not missing, f"{len(need) - len(missing)}/{len(need)} present under {d}"):
            continue
        npz = np.load(d / "effects.npz", allow_pickle=False)
        pert = pd.read_parquet(d / "perturbations.parquet")
        genes = pd.read_parquet(d / "genes.parquet")
        ceil = pd.read_parquet(d / "noise_ceiling_summary.parquet")
        qc = json.loads((d / "qc.json").read_text())

        n_t, n_g = npz["lfc_within_batch"].shape
        ch.add(sec, "shapes", n_t == pert.shape[0] and n_g == genes.shape[0],
               f"{n_t} perturbations x {n_g} genes, tables agree")
        mats = ["delta_within_batch", "delta_pooled", "lfc_within_batch", "lfc_pooled"]
        bad = {m: int((~np.isfinite(npz[m])).sum()) for m in mats}
        ch.add(sec, "finite", all(v == 0 for v in bad.values()), f"non-finite entries {bad}")
        ch.add(sec, "min_cells", int(pert.n_cells.min()) >= C.PERTURBATION_FILTERS.min_cells,
               f"smallest retained perturbation has {int(pert.n_cells.min())} cells "
               f"(threshold {C.PERTURBATION_FILTERS.min_cells})")
        kd = pert.knockdown.dropna()
        ch.add(sec, "knockdown", float(kd.median()) > 0.5,
               f"median on-target knockdown {float(kd.median()):.4f} over {kd.size} measured targets")
        ch.add(sec, "de_counts", int(pert.n_de_genes.sum()) > 0,
               f"median {float(pert.n_de_genes.median()):.1f} DE genes per perturbation, "
               f"max {int(pert.n_de_genes.max())}")
        strata = pert.effect_stratum.value_counts().to_dict()
        ch.add(sec, "strata", len(strata) == C.EFFECTS.n_strata,
               f"{len(strata)} strata {dict(sorted(strata.items()))}")
        ch.add(sec, "expressed", int(genes.expressed.sum()) > 0,
               f"{int(genes.expressed.sum())} genes above {C.EFFECTS.expressed_min_cp10k} CP10K")
        retained = qc.get("retained", {})
        ch.add(
            sec,
            "qc_json",
            {"cells", "control_cells", "perturbations"} <= set(retained)
            and "config" in qc
            and "source" in qc,
            f"{retained.get('cells')} cells retained, {retained.get('control_cells')} of them "
            f"non-targeting controls, over {retained.get('perturbations')} perturbations; "
            f"provenance and config recorded",
        )

        # noise ceilings: bounded, and higher for larger effects
        in_range = ceil[(ceil["median"] <= 0) | (ceil["median"] > 1)]
        ch.add(sec, "ceiling_range", in_range.empty,
               f"{ceil.shape[0]} ceiling estimates all in (0, 1]" if in_range.empty
               else f"{in_range.shape[0]} estimates outside (0, 1]: "
                    + str(in_range[["split", "metric", "stratum", "median"]].to_dict("records")[:4]))
        for split in C.CEILING_SPLITS:
            sub = ceil[(ceil.split == split) & (ceil.metric == "pearson_lfc")]
            if sub.empty:
                ch.add(sec, f"ceiling_{split}", False, "no pearson_lfc rows for this split")
                continue
            by = sub.set_index("stratum")["median"]
            top, bot = by.get(C.STRATUM_NAMES[-1]), by.get(C.STRATUM_NAMES[0])
            ch.add(sec, f"ceiling_{split}", top is not None and bot is not None and top > bot,
                   f"pearson_lfc {C.STRATUM_NAMES[0]}={bot:.4f} < {C.STRATUM_NAMES[-1]}={top:.4f}"
                   if top is not None and bot is not None else "strata missing")

        check_ceiling_grid(ch, d, cl, pert, genes)


def check_ceiling_grid(
    ch: Checks, d: Path, cl: str, pert: pd.DataFrame, genes: pd.DataFrame
) -> None:
    """The top-k overlap grid: k values, chance levels, adaptive k, persisted vectors.

    The grid is a recompute of one metric of the ceiling stage, so the first thing
    checked is that it reproduces the ceiling columns already published rather than
    a second, slightly different measurement.
    """
    sec = f"grid:{cl}"
    need = [
        "topk_overlap_grid.parquet",
        "topk_overlap_grid_summary.parquet",
        "topk_overlap_grid_qc.json",
        "split_half_vectors.npz",
    ]
    missing = [f for f in need if not (d / f).exists()]
    if not ch.add(sec, "files", not missing,
                  f"missing {missing}" if missing else f"{len(need)} files under {d}"):
        return
    grid = pd.read_parquet(d / "topk_overlap_grid.parquet")
    summary = pd.read_parquet(d / "topk_overlap_grid_summary.parquet")
    qc = json.loads((d / "topk_overlap_grid_qc.json").read_text())
    n_expressed = int(genes.expressed.sum())

    repro = qc["reproduction_of_published_ceilings"]["max_abs_diff"]
    worst = max(repro.values())
    ch.add(sec, "reproduces_ceilings", worst == 0.0,
           f"recompute reproduces every published ceiling column exactly "
           f"(max abs diff {worst:g} over {len(repro)} split x metric combinations, "
           f"pooled control CP10K {qc['reproduction_of_published_ceilings']['max_abs_diff_pooled_control_cp10k']:g})")

    worst_k50 = 0.0
    for split in C.CEILING_SPLITS:
        g = grid[(grid.split == split) & (grid.k_label == str(C.EFFECTS.topk))]
        merged = g.merge(
            pert[["target", f"{split}__topk_overlap"]], on="target", how="left"
        )
        worst_k50 = max(
            worst_k50, float((merged.overlap - merged[f"{split}__topk_overlap"]).abs().max())
        )
    ch.add(sec, "k50_column_unchanged", worst_k50 == 0.0,
           f"the k={C.EFFECTS.topk} column of the grid is the published topk_overlap "
           f"column, both splits (max abs diff {worst_k50:g} over {pert.shape[0]:,} perturbations)")

    expect = grid.k.to_numpy(dtype=float) / n_expressed
    ch.add(sec, "chance_level", bool(np.allclose(grid.chance.to_numpy(dtype=float), expect)),
           f"chance = k/{n_expressed:,} expressed genes: "
           + ", ".join(f"k={k} -> {k / n_expressed:.4f}" for k in K_GRID))

    fin = grid[np.isfinite(grid.overlap.to_numpy(dtype=float))]
    ex = (fin.overlap - fin.chance) / (1.0 - fin.chance)
    ch.add(sec, "excess_over_chance",
           bool(np.allclose(fin.excess_over_chance.to_numpy(dtype=float), ex.to_numpy())
                and (fin.excess_over_chance > 0).mean() > 0.9),
           f"(overlap - chance)/(1 - chance) on {fin.shape[0]:,} rows, "
           f"{100 * float((fin.excess_over_chance > 0).mean()):.1f}% above chance")

    # At k = 10 the per-perturbation overlap can only take the eleven values
    # 0, 0.1, ... 1, so the median is quantised and the strata can tie; the
    # ordering claim is therefore strict only for the k values that can resolve
    # it, and non-decreasing at k = 10.
    labels = [str(k) for k in K_GRID] + ["adaptive"]
    order_ok, detail = True, []
    for split in C.CEILING_SPLITS:
        for label in labels:
            s = summary[(summary.split == split) & (summary.k_label == label)].set_index("stratum")
            lo = float(s.loc[C.STRATUM_NAMES[0], "median_excess_over_chance"])
            hi = float(s.loc[C.STRATUM_NAMES[-1], "median_excess_over_chance"])
            order_ok &= hi >= lo if label == "10" else hi > lo
            if split == "split_half_cells":
                detail.append(f"k={label}: {lo:.4f} -> {hi:.4f}")
    ch.add(sec, "excess_rises_with_effect", order_ok,
           f"{C.STRATUM_NAMES[0]} -> {C.STRATUM_NAMES[-1]} excess over chance, "
           f"split by cell ({'; '.join(detail)}); strict for k >= "
           f"{K_GRID[1]}, non-decreasing at k = {K_GRID[0]} where the median is "
           f"quantised to steps of {1 / K_GRID[0]:g}")

    ad = grid[grid.k_label == "adaptive"]
    want = np.clip(ad.n_de_genes.to_numpy(dtype=np.int64), ADAPTIVE_K_MIN, ADAPTIVE_K_MAX)
    ch.add(sec, "adaptive_k", bool((ad.k.to_numpy(dtype=np.int64) == want).all()),
           f"k = clip(n_de_genes, {ADAPTIVE_K_MIN}, {ADAPTIVE_K_MAX}) on every row; "
           f"median k {float(ad.k.median()):.0f}, "
           f"{100 * float((ad.k == ADAPTIVE_K_MIN).mean()):.1f}% at the floor and "
           f"{100 * float((ad.k == ADAPTIVE_K_MAX).mean()):.1f}% at the ceiling")

    floor_ok = bool((fin.tie_fraction >= 1.0 / fin.k - 1e-9).all() and (fin.tie_fraction <= 1).all())
    contested = float((fin.n_tied_excluded > 0).mean())
    ch.add(sec, "selection_diagnostics",
           floor_ok and bool((fin.median_control_cp10k_selected > 0).all()),
           f"tie fraction within [1/k, 1] everywhere; {100 * contested:.2f}% of selections "
           f"have any gene tied with the k-th value; median control level of the selected "
           f"genes {float(fin.median_control_cp10k_selected.median()):.4f} CP10K against "
           f"{float(genes[genes.expressed].pooled_control_cp10k.median()):.4f} over all expressed genes")

    npz = np.load(d / "split_half_vectors.npz", allow_pickle=False)
    arrays = [k for k in npz.files if npz[k].ndim == 2]
    shape_ok = all(npz[k].shape == (pert.shape[0], n_expressed) for k in arrays)
    targets_ok = list(npz["targets"]) == list(pert.target)
    rng = np.random.default_rng(C.SEED)
    sample = rng.choice(pert.shape[0], size=min(25, pert.shape[0]), replace=False)
    worst_vec = 0.0
    for split in C.CEILING_SPLITS:
        a0, a1 = npz[f"{split}__lfc_side0"], npz[f"{split}__lfc_side1"]
        ref = pert[f"{split}__topk_overlap"].to_numpy(dtype=float)
        k = C.EFFECTS.topk
        for i in sample:
            s0 = set(np.argpartition(-np.abs(a0[i]), k - 1)[:k].tolist())
            s1 = set(np.argpartition(-np.abs(a1[i]), k - 1)[:k].tolist())
            if np.isfinite(ref[i]):
                worst_vec = max(worst_vec, abs(len(s0 & s1) / k - ref[i]))
    ch.add(sec, "persisted_vectors",
           len(arrays) == 8 and shape_ok and targets_ok and worst_vec <= 0.02,
           f"{len(arrays)} float32 arrays of {pert.shape[0]:,} x {n_expressed:,} "
           f"({(d / 'split_half_vectors.npz').stat().st_size / 1e6:.0f} MB); rows follow "
           f"perturbations.parquet; overlap recomputed from them differs from the "
           f"published column by at most {worst_vec:g} over {sample.size} sampled perturbations")


def check_crosscheck(ch: Checks, root: Path) -> None:
    p = root / "crosscheck" / "summary.json"
    if not ch.add("crosscheck", "files", p.exists(), f"{p}"):
        return
    s = json.loads(p.read_text())
    rho, n = s.get("pooled_spearman", float("nan")), s.get("n_pooled_entries", 0)
    ch.add("crosscheck", "pooled_spearman",
           isinstance(rho, float) and not math.isnan(rho) and rho >= C.CROSSCHECK_MIN_SPEARMAN,
           f"Spearman {rho:.4f} vs the tahoebio reference over {n:,} "
           f"(cell line, target, gene) log2FC entries; gate >= {C.CROSSCHECK_MIN_SPEARMAN}")
    ch.add("crosscheck", "pairs", s.get("n_shared_cell_line_target_pairs", 0) > 100,
           f"{s.get('n_shared_cell_line_target_pairs', 0)} shared perturbations, median "
           f"per-perturbation Spearman {s.get('median_per_pair_spearman', float('nan')):.4f}")
    own = s.get("own_estimator_agreement", {})
    ch.add("crosscheck", "own_estimators",
           own.get("median_pearson_within_vs_pooled_lfc") is not None,
           f"batch-matched vs batch-pooled log2FC median Pearson "
           f"{own.get('median_pearson_within_vs_pooled_lfc', float('nan')):.4f}")


def check_universe(ch: Checks, root: Path) -> None:
    d = root / "gene_universe"
    if not ch.add("universe", "files", (d / "gene_universe.parquet").exists(), str(d)):
        return
    uni = pd.read_parquet(d / "gene_universe.parquet")
    cov = json.loads((d / "coverage.json").read_text())
    ch.add("universe", "size", uni.shape[0] > 5000,
           f"{uni.shape[0]:,} genes present in all four model vocabularies "
           f"(of {cov.get('shared_measured_genes')} measured in both cell lines)")
    ch.add("universe", "unique", uni.gene_id.is_unique, f"{uni.gene_id.nunique():,} unique Ensembl ids")
    for m in MODELS:
        col = f"{m}_position"
        v = pd.read_parquet(d / f"vocabulary_{m}.parquet")
        pos = uni[col].to_numpy()
        ch.add("universe", f"positions_{m}",
               bool((pos >= 0).all() and (pos < v.shape[0]).all()),
               f"positions within [0, {v.shape[0]:,}) and unique={uni[col].is_unique}")
        vc = cov["coverage"][m]
        ch.add("universe", f"coverage_{m}", vc["covers_hepg2_measured"] > 5000,
               f"covers {vc['covers_hepg2_measured']:,}/{vc['hepg2_measured_total']:,} HepG2 and "
               f"{vc['covers_jurkat_measured']:,}/{vc['jurkat_measured_total']:,} Jurkat measured "
               f"genes; {vc['n_unmapped_tokens']:,} of {vc['vocabulary_size']:,} vocabulary tokens "
               f"unmapped; identifiers are {vc['identifier_kind']}, "
               f"order_is_a_model_constraint={vc['order_is_a_model_constraint']}")
    audit = cov.get("identifier_audit", {})
    fwd = audit.get("ensg_to_symbol_to_ensg_failures")
    rev = audit.get("symbol_to_ensg_to_symbol_failures")
    n = audit.get("n_nadig_genes")
    ch.add("universe", "round_trip", fwd is not None and rev is not None and fwd + rev <= 2,
           f"{fwd} Ensembl->symbol->Ensembl and {rev} symbol->Ensembl->symbol failures over "
           f"{n:,} Nadig genes ({audit.get('symbols_not_hgnc_approved')} symbols not HGNC-approved); "
           f"examples {audit.get('round_trip_failure_examples')}")


def check_models(ch: Checks, root: Path) -> None:
    for m in MODELS:
        d = root / "models" / m
        sec = f"model:{m}"
        need = ["response_vector.parquet", "contract.json", "environment.json", "demo.json"]
        missing = [f for f in need if not (d / f).exists()]
        if not ch.add(sec, "files", not missing, f"missing {missing}" if missing else f"all present under {d}"):
            continue
        resp = pd.read_parquet(d / "response_vector.parquet")
        con = json.loads((d / "contract.json").read_text())
        demo = json.loads((d / "demo.json").read_text())
        env = json.loads((d / "environment.json").read_text())
        uni_n = pd.read_parquet(root / "gene_universe" / "gene_universe.parquet").shape[0]
        ch.add(sec, "response", resp.shape[0] == uni_n and np.isfinite(resp.response).all(),
               f"{resp.shape[0]:,} finite per-gene values over the universe")
        for field in ("input_normalization", "output_head", "output_semantics", "information_flow"):
            ch.add(sec, f"contract_{field}", bool(str(con.get(field, "")).strip()),
                   f"{len(str(con.get(field, '')))} characters recorded")
        # symmetry must be either answered or explicitly measured-and-inconclusive:
        # a silent None would let a later run assume a symmetric response matrix
        probe = demo.get("jacobian", {}).get("symmetry_probe") or {}
        answered = con.get("response_symmetric") is not None
        explained = probe.get("verdict") in ("not_determined", "not_measured") and bool(
            probe.get("note") or probe.get("reason")
        )
        ch.add(
            sec,
            "symmetry",
            (answered or explained) and bool(str(con.get("symmetry_evidence", "")).strip()),
            f"response_symmetric={con.get('response_symmetric')}, "
            f"ordering_dependent={con.get('ordering_dependent')}, "
            f"cross-entry probe {probe.get('verdict', 'absent')}"
            + (
                f" (ratio {probe['ratio_d_source_d_target_over_d_target_d_source']:.4g})"
                if "ratio_d_source_d_target_over_d_target_d_source" in probe
                else ""
            ),
        )
        jac = demo.get("jacobian", {})
        ok = jac.get("available") or bool(jac.get("unavailable_reason"))
        ch.add(sec, "jacobian", bool(ok),
               f"available={jac.get('available')} method={jac.get('method')} "
               f"l2={jac.get('column_l2')}" if jac.get("available")
               else f"unavailable: {jac.get('unavailable_reason')}")
        if jac.get("available"):
            ch.add(sec, "jacobian_file", (d / "jacobian_column.parquet").exists(),
                   "jacobian_column.parquet written")
        ch.add(sec, "environment", len(env.get("pinned_packages", [])) > 5,
               f"{len(env.get('pinned_packages', []))} pinned packages, image "
               f"{env.get('image')}, device {env.get('device')}")
        subs = con.get("substitutions") or []
        ch.add(sec, "substitutions", True,
               f"{len(subs)} recorded substitution(s)" + (f": {subs[0][:90]}" if subs else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="run every section")
    ap.add_argument("--effects", action="store_true")
    ap.add_argument("--crosscheck", action="store_true")
    ap.add_argument("--universe", action="store_true")
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--root", default=None, help="artifact root to verify (default: SCFM_PFM_DATA_ROOT)")
    args = ap.parse_args(argv)
    sections = {
        "effects": args.all or args.effects,
        "crosscheck": args.all or args.crosscheck,
        "universe": args.all or args.universe,
        "models": args.all or args.models,
    }
    if not any(sections.values()):
        sections = dict.fromkeys(sections, True)
    root = Path(args.root) if args.root else paths.input_root()
    print(f"verifying artifacts under {root}\n")
    ch = Checks()
    if sections["effects"]:
        check_effects(ch, root)
    if sections["universe"]:
        check_universe(ch, root)
    if sections["crosscheck"]:
        check_crosscheck(ch, root)
    if sections["models"]:
        check_models(ch, root)
    print(ch.report())
    n_fail = len(ch.failed)
    print(f"\n{len(ch.rows) - n_fail}/{len(ch.rows)} checks passed")
    if n_fail:
        print("FAILED:")
        for section, name, _, detail in ch.failed:
            print(f"  {section}/{name}: {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
