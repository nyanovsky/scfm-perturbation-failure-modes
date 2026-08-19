"""The verification contract for run 2's feature blocks.

Two modes, because the arrays and the report live in different places on this
fabric. ``--block-dir`` runs where a block's arrays are (inside a job that mounts
the producing job's volume): it opens every npz, checks shapes against the
recorded index tables, checks finiteness, measures the softmax coupling, and
writes two files beside the block --- ``verification.json`` (the checks) and
``report_bundle.json`` (everything the pod-side report needs, small enough to
travel). ``--all`` runs on the pod over the collected bundles plus the
floor-sweep outputs, and prints the whole contract.

    python -m scfm_pfm.verify_features --block-dir <features/<model>/<line>>
    python -m scfm_pfm.verify_features --all --bundles <dir of report_bundle*.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import paths

#: a softmax model's per-query response differences must sum to about zero
SOFTMAX_MEAN_TOLERANCE = 0.05  # |cross-gene mean| / cross-gene sd
#: exact derivative against its finite-difference cross-check
EXACT_VS_FD_MIN_PEARSON = 0.99
EXACT_VS_FD_MAX_RELATIVE_L2 = 0.25
SOFTMAX_MODELS = ("scprint", "transcriptformer")
FORMS = ("zero_out", "vvp", "gdt")


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, section: str, name: str, ok: bool, detail: str) -> bool:
        self.rows.append({"section": section, "check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {section}/{name}: {detail}", flush=True)
        return bool(ok)

    def summary(self) -> tuple[int, int]:
        ok = sum(1 for r in self.rows if r["ok"])
        return ok, len(self.rows)


# --------------------------------------------------------------------------- #
# block-level: runs where the arrays are
# --------------------------------------------------------------------------- #
def verify_block(block_dir: Path, ch: Checks) -> dict:
    manifest = json.loads((block_dir / "manifest.json").read_text())
    model, cell_line = manifest["model"], manifest["cell_line"]
    sec = f"{model}/{cell_line}"
    readout = pd.read_parquet(block_dir / "readout_genes.parquet")
    perturbed = pd.read_parquet(block_dir / "perturbed_genes.parquet")
    n_r, n_p = readout.shape[0], perturbed.shape[0]
    n_v = len(manifest["value_grid_cp10k"])
    bundle: dict = {
        "model": model,
        "cell_line": cell_line,
        "manifest": manifest,
        "n_readout": int(n_r),
        "n_perturbed": int(n_p),
        "blocks": {},
        "query_summary": {},
        "step_sweep": [],
    }

    expected_rows = {
        "uniform_virtual": n_p,
        "mean_control": n_p,
        "individual_cells": int(perturbed.in_cells_anchor.sum()),
        "mean_control_masked": int(perturbed.in_masked.sum()),
    }
    for adir in sorted(block_dir.glob("anchor=*")):
        anchor = adir.name.removeprefix("anchor=")
        want_p = expected_rows.get(anchor, n_p)
        masked = anchor.endswith("_masked")
        forms = ("zero_out", "vvp") if masked else FORMS
        for form in forms:
            path = adir / f"{form}.npz"
            if not ch.add(sec, f"{anchor}/{form}/exists", path.exists(), str(path.name)):
                continue
            z = np.load(path, allow_pickle=False)
            n_vals = len(z["value_grid_cp10k"]) if "value_grid_cp10k" in z.files else n_v
            want = (want_p, n_r) if form == "zero_out" else (want_p, n_vals, n_r)
            raw = z["raw"]
            shapes_ok = raw.shape == want
            finite = bool(np.isfinite(raw).all())
            ch.add(
                sec,
                f"{anchor}/{form}/shape_and_finite",
                shapes_ok and finite,
                f"raw {raw.shape} (expected {want}), finite={finite}, "
                f"variants {sorted(k for k in z.files if k in C.NORMALIZATIONS)}",
            )
            have_recentred = "recentred" in z.files
            if model in SOFTMAX_MODELS:
                ch.add(
                    sec,
                    f"{anchor}/{form}/recentred_present",
                    have_recentred,
                    "the recentred variant sits beside the raw one for a softmax model",
                )
            # A response that is identically zero is a query the model could not see:
            # the perturbed value landed on the value the input already carried. It is a
            # real property of a quantized input, and a consumer needs to know how much
            # of a block is degenerate before averaging over it.
            flat = raw.reshape(-1, raw.shape[-1])
            bundle["blocks"][f"{anchor}/{form}"] = {
                "shape": list(raw.shape),
                "finite": finite,
                "variants": sorted(k for k in z.files if k in C.NORMALIZATIONS),
                "abs_median": float(np.median(np.abs(raw))),
                "abs_p99": float(np.percentile(np.abs(raw), 99)),
                "all_zero_fraction": float((np.abs(flat).max(axis=1) == 0).mean()),
                "n_response_vectors": int(flat.shape[0]),
            }
        qs = adir / "query_summary.parquet"
        if qs.exists():
            frame = pd.read_parquet(qs)
            # How much the choice of anchor cell moves the size of the response. The
            # block stores responses averaged over cells, so this is the only handle on
            # it: the spread of the per-cell response size across cells, relative to its
            # own mean, for the same perturbed gene and value.
            if "cell_index" in frame.columns and frame.cell_index.nunique() > 1:
                per_cell = frame.groupby(["perturbed_gene", "value_cp10k"])["cross_gene_sd"]
                mean = per_cell.transform("mean")
                cv = (per_cell.transform("std") / mean.where(mean > 0)).dropna()
                bundle.setdefault("anchor_cell_spread", {})[anchor] = {
                    "n_cells": int(frame.cell_index.nunique()),
                    "median_cv_of_response_size": float(cv.median()),
                    "p90_cv_of_response_size": float(cv.quantile(0.9)),
                }
            bundle["query_summary"][anchor] = (
                frame.groupby("value_cp10k")
                .agg(
                    cross_gene_mean=("cross_gene_mean", "median"),
                    cross_gene_sd=("cross_gene_sd", "median"),
                    max_abs=("max_abs", "median"),
                    self_delta=("self_delta", "median"),
                    self_rank_median=("self_rank_by_abs", "median"),
                    n_queries=("cross_gene_mean", "size"),
                    # a query the model could not see: the value asked for was the value
                    # the input already carried, so the two passes were the same call
                    n_no_op_queries=("max_abs", lambda s: int((s == 0).sum())),
                )
                .reset_index()
                .to_dict(orient="records")
            )
            if model in SOFTMAX_MODELS:
                ratio = float(
                    np.median(np.abs(frame.cross_gene_mean) / np.maximum(frame.cross_gene_sd, 1e-30))
                )
                ch.add(
                    sec,
                    f"{anchor}/softmax_coupling",
                    ratio <= SOFTMAX_MEAN_TOLERANCE,
                    f"median |cross-gene mean| / sd = {ratio:.4g} (tolerance "
                    f"{SOFTMAX_MEAN_TOLERANCE}); raising one gene's share must lower others",
                )

    # coverage: how much of the in-scope perturbation set this block covers. Recorded
    # explicitly because a subset block and a full block are not interchangeable and
    # the difference is invisible in anything but an array shape.
    cov = manifest.get("coverage")
    if cov:
        ch.add(
            sec,
            "coverage_recorded",
            cov.get("n_perturbed") == n_p
            and cov.get("n_in_scope", 0) >= n_p
            and cov.get("mode") in ("full", "subset"),
            f"{cov.get('mode')} coverage: {cov.get('n_perturbed')} of "
            f"{cov.get('n_in_scope')} in-scope perturbations "
            f"({cov.get('fraction_of_scope')}), strata {cov.get('strata')}; "
            f"perturbed table has {n_p} rows",
        )
    else:
        ch.add(
            sec,
            "coverage_recorded",
            False,
            f"the manifest records no coverage block, so this block's {n_p} perturbed genes "
            "cannot be placed against the in-scope set",
        )

    # step selection and its evidence
    steps = manifest.get("step_selection", {})
    complete = all(
        "selected_step" in steps.get(a, {}) and steps[a].get("adjacent_agreement")
        for a in manifest["anchors"]
    )
    flagged = [a for a, s in steps.items() if s.get("verdict") == "magnitude_unstable"]
    ch.add(
        sec,
        "step_selection",
        complete,
        "selected step and adjacent-step agreement recorded for every anchor; "
        f"magnitude-unstable anchors: {flagged or 'none'}",
    )
    if (block_dir / "step_sweep.parquet").exists():
        bundle["step_sweep"] = pd.read_parquet(block_dir / "step_sweep.parquet").to_dict(
            orient="records"
        )

    # exact derivative against its finite-difference cross-check, per value: at an
    # input value on the floor the difference is one-sided and is reported separately
    for anchor, meta in manifest["anchors"].items():
        chk = meta.get("exact_vs_finite_difference")
        if not chk:
            continue
        if "per_value" in chk:
            pear = chk.get("median_pearson_two_sided")
            rel = chk.get("median_relative_l2_two_sided")
            one_sided = [r for r in chk["per_value"] if r["one_sided"]]
            ok = (
                pear is not None
                and pear >= EXACT_VS_FD_MIN_PEARSON
                and (rel or 0) <= EXACT_VS_FD_MAX_RELATIVE_L2
            )
            detail = (
                f"median over two-sided values: pearson {pear:.6f}, relative L2 {rel:.4g} at "
                f"step {chk['step']} (tolerances {EXACT_VS_FD_MIN_PEARSON}, "
                f"{EXACT_VS_FD_MAX_RELATIVE_L2}); "
                + "; ".join(
                    f"{r['value_cp10k']:g} CP10K {'one-sided ' if r['one_sided'] else ''}"
                    f"pearson {r['pearson']:.4f}" if r["pearson"] is not None else
                    f"{r['value_cp10k']:g} CP10K undefined"
                    for r in chk["per_value"]
                )
            )
            if one_sided:
                detail += (
                    f"; {len(one_sided)} value(s) sit on the input floor, where the difference "
                    "is one-sided and is excluded from the medians"
                )
        else:
            ok = (
                chk["pearson"] >= EXACT_VS_FD_MIN_PEARSON
                and (chk.get("relative_l2") or 0) <= EXACT_VS_FD_MAX_RELATIVE_L2
            )
            detail = (
                f"pearson {chk['pearson']:.6f}, relative L2 {chk.get('relative_l2'):.4g} at step "
                f"{chk['step']}"
            )
        ch.add(sec, f"{anchor}/exact_vs_finite_difference", ok, detail)

    if manifest.get("untiled_exact_check", {}).get("ran"):
        u = manifest["untiled_exact_check"]
        ch.add(
            sec,
            "untiled_exact_derivative",
            u["median_pearson_two_sided"] >= EXACT_VS_FD_MIN_PEARSON,
            f"in the untiled layout the exact forward-mode derivative agrees with the "
            f"central difference at step {u['step']} over {u['n_genes']} genes: median "
            f"pearson {u['median_pearson_two_sided']:.6f}, relative L2 "
            f"{u['median_relative_l2_two_sided']:.4g}",
        )

    # wiring: where the guard does not reach the threshold, say what it is tracking
    detail_rows = wiring_detail(block_dir, manifest, readout, perturbed)
    if detail_rows:
        bundle["wiring_detail"] = detail_rows
        ch.add(
            sec,
            "wiring_by_anchor_level",
            True,
            "own-entry-largest rate by the perturbed gene's own anchor level: "
            + "; ".join(
                f"{r['bin']} ({r['n']} genes) {r['pass_fraction']:.3f}" for r in detail_rows
            )
            + f"; the value grid tops out at {max(C.VALUE_GRID_CP10K):g} CP10K, so a gene "
            "whose anchor sits far above it is barely moved by the query",
        )

    # alignment: the pre-registered guard plus two statistics that keep their power
    # when it structurally cannot have any
    align_rows = alignment_detail(block_dir, manifest, perturbed)
    if align_rows:
        bundle["alignment"] = align_rows
        for r in align_rows:
            ch.add(
                sec,
                f"{r['anchor']}/alignment_vs_permutation",
                r["aligned"],
                "read-out order is aligned with the query on "
                + (", ".join(r["separated_by"]) or "no statistic")
                + f": own entry largest {r['own_largest']:.3f} (null {r['own_largest_null']:.3f}); "
                f"median self-rank {r['self_rank_median']:.0f} of {r['n_readout']} "
                f"(null {r['self_rank_median_null']:.0f}); own output moves the expected way "
                f"{r['sign_rate']:.3f} (null {r['sign_rate_null']:.3f})",
            )

    for anchor, w in manifest.get("wiring_checks", {}).items():
        ch.add(
            sec,
            f"{anchor}/wiring_magnitude",
            w["magnitude_check_pass_fraction"] >= C.SELF_RESPONSE_MIN_PASS,
            f"the perturbed gene's own entry is the largest response for "
            f"{100 * w['magnitude_check_pass_fraction']:.1f}% of {w['n_genes']} sampled genes "
            f"(threshold {100 * C.SELF_RESPONSE_MIN_PASS:.0f}%); sign version "
            f"{100 * w['sign_check_pass_fraction']:.1f}%",
        )
        ch.add(
            sec,
            f"{anchor}/wiring_permutation",
            w["permuted_magnitude_pass_fraction"] < 0.05,
            f"shuffling the read-out order drops the same test to "
            f"{100 * w['permuted_magnitude_pass_fraction']:.1f}%, so the guard has power",
        )

    # supplied set, one per (model, cell line), identical across queries
    layout = manifest["layout"]
    ch.add(
        sec,
        "supplied_set_recorded",
        layout.get("n_supplied_genes", 0) > 0 and layout.get("supplied_mode"),
        f"{layout['n_supplied_genes']} genes, mode {layout['supplied_mode']}, "
        f"{layout['forward_passes_per_query']} pass(es) per query; fixed for every query "
        "in this (model, cell line)",
    )

    # model-specific contracts
    if model == "transcriptformer":
        ok = all(
            b["finite"] for k, b in bundle["blocks"].items()
        )
        ch.add(
            sec,
            "transcriptformer/ordering",
            ok and layout.get("context_block_genes") == 256,
            f"perturbed-gene-first ordering gave finite features for every pair; shared "
            f"{layout.get('context_block_genes')}-gene prefix; per-pass shares normalized by "
            f"the pass total; {layout.get('unmeasurable_fraction_under_fixed_ordering')} of "
            "ordered pairs would be unmeasurable under one fixed ordering",
        )
    if model == "scgpt" and manifest.get("context_comparison"):
        cc = manifest["context_comparison"]
        ch.add(
            sec,
            "scgpt/context_comparison",
            bool(cc.get("agreement")) and cc.get("production_mode"),
            f"{cc['n_genes']} common genes; "
            + "; ".join(
                f"{k} pearson {v['median_pearson']:.4f} relative L2 {v['median_relative_l2']:.4f}"
                for k, v in cc["agreement"].items()
            )
            + f"; production width {cc['production_mode']}",
        )
    if model in SOFTMAX_MODELS:
        pass

    # masked variant
    masked = manifest.get("masked", {})
    if masked.get("ran"):
        cov = masked["coverage_per_gene"]
        ch.add(
            sec,
            "masked_coverage",
            cov["min"] >= C.MASK_MIN_COVERAGE,
            f"mask rate {masked['mask_rate']} in {masked['n_blocks_per_cycle']} disjoint blocks, "
            f"{masked['cycles']} cycles; every read-out gene read masked "
            f"{cov['min']}-{cov['max']} times",
        )
    else:
        ch.add(
            sec,
            "masked_variant",
            bool(masked.get("reason")),
            f"no masked variant: {masked.get('reason', 'unrecorded')}",
        )

    # the manifest itself
    contract = manifest.get("contract", {})
    fields = {
        "checkpoint": contract.get("checkpoint"),
        "checkpoint_revision": contract.get("checkpoint_revision"),
        "anchors": sorted(manifest["anchors"]),
        "value_grid_cp10k": manifest["value_grid_cp10k"],
        "expression_floor_cp10k": manifest.get("expression_floor_cp10k"),
        "n_substitutions": len(manifest.get("substitutions", [])),
    }
    ch.add(
        sec,
        "manifest_complete",
        all(v not in (None, [], "") for v in fields.values()),
        json.dumps(fields, default=str),
    )
    bundle["checks"] = [r for r in ch.rows if r["section"] == sec]
    return bundle


def wiring_detail(block_dir: Path, manifest: dict, readout: pd.DataFrame, perturbed: pd.DataFrame) -> list[dict]:
    """Does the wiring guard fail where the query barely moves the gene?

    The guard reads the response between the lowest positive value on the grid and
    the highest. A perturbed gene whose own anchor level sits far above the top of
    the grid is hardly perturbed at all by that comparison, so its own entry need
    not be the largest response even when the wiring is right. This splits the
    pass rate by the perturbed gene's anchor level to show which case a failure is.
    """
    path = block_dir / "anchor=mean_control" / "vvp.npz"
    if not path.exists():
        return []
    z = np.load(path, allow_pickle=False)
    raw = z["raw"]
    grid = list(z["value_grid_cp10k"]) if "value_grid_cp10k" in z.files else list(C.VALUE_GRID_CP10K)
    positive = [i for i, v in enumerate(grid) if v > 0]
    lo_i, hi_i = positive[0], positive[-1]
    cols = perturbed_cols(z, perturbed)
    level = readout["pooled_control_cp10k"].to_numpy(dtype=np.float64)
    passes, levels = [], []
    for pi, col in enumerate(cols):
        mag = np.abs(raw[pi, hi_i] - raw[pi, lo_i])
        passes.append(int((mag > mag[int(col)]).sum()) == 0)
        levels.append(float(level[int(col)]))
    passes, levels = np.array(passes), np.array(levels)
    top = float(max(grid))
    bins = [
        ("at or below the top of the value grid", levels <= top),
        ("above the top of the value grid", levels > top),
    ]
    return [
        {
            "bin": name,
            "n": int(mask.sum()),
            "pass_fraction": float(passes[mask].mean()) if mask.any() else float("nan"),
            "median_anchor_cp10k": float(np.median(levels[mask])) if mask.any() else float("nan"),
        }
        for name, mask in bins
        if mask.any()
    ]


def perturbed_cols(z: "np.lib.npyio.NpzFile", perturbed: pd.DataFrame) -> np.ndarray:
    """The read-out column of each row of an array, read from the array's own labels.

    An anchor may run over a subset of the perturbed genes --- the individual-cell
    anchor does --- and that subset is not the first rows of the perturbed table.
    Every array file stores the gene ids of its own rows, so join on those rather
    than assuming an order.
    """
    lookup = dict(
        zip(perturbed["gene_id"].astype(str), perturbed["readout_col"].astype(int))
    )
    if "perturbed_gene_id" in z.files:
        ids = [str(g) for g in z["perturbed_gene_id"]]
        missing = [g for g in ids if g not in lookup]
        if missing:
            raise SystemExit(f"{len(missing)} array rows are not in the perturbed table: {missing[:3]}")
        return np.asarray([lookup[g] for g in ids], dtype=np.int64)
    return perturbed["readout_col"].to_numpy(dtype=np.int64)


#: an alignment statistic counts as separated from its permuted null by these margins
ALIGN_RANK_MAX_FRACTION = 0.05  # own entry inside the top 5% of the response vector
ALIGN_RANK_NULL_MIN_FRACTION = 0.25  # ... while a relabelled column sits mid-vector
ALIGN_SIGN_MIN = 0.80  # own output moves the expected way this often
ALIGN_SIGN_NULL_MAX = 0.65  # ... while a relabelled column is near a coin flip


def alignment_detail(block_dir: Path, manifest: dict, perturbed: pd.DataFrame) -> list[dict]:
    """Three readings of the same alignment question, each against its permuted null.

    The pre-registered guard asks whether the perturbed gene's own entry is the
    *largest* response. That question has no power for an autoregressive model:
    with the perturbed gene placed first it never conditions on its own value, so
    its own entry is never the largest and the permuted control is zero too ---
    both the test and its null sit at the floor, and the comparison says nothing.

    So record three statistics per anchor, each with the null obtained by reading
    a randomly relabelled column instead of the true one:

    ``own_largest``  fraction whose own entry is the largest response (null ~0)
    ``self_rank``    median rank of the own entry among read-out genes (null ~n/2)
    ``sign``         fraction whose own output moves the expected way (null ~0.5)

    A misalignment would drive all three to their nulls at once, so one statistic
    that separates widely is enough to rule it out.
    """
    rows = []
    rng = np.random.default_rng(C.SEED)
    for anchor in manifest.get("anchors", []):
        path = block_dir / f"anchor={anchor}" / "vvp.npz"
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=False)
        raw = z["raw"]
        grid = (
            list(z["value_grid_cp10k"]) if "value_grid_cp10k" in z.files else list(C.VALUE_GRID_CP10K)
        )
        positive = [i for i, v in enumerate(grid) if v > 0]
        lo_i, hi_i = positive[0], positive[-1]
        n_p, n_r = raw.shape[0], raw.shape[2]
        cols = perturbed_cols(z, perturbed)
        if cols.shape[0] != n_p:
            raise SystemExit(f"{anchor}: {cols.shape[0]} labelled rows for {n_p} array rows")
        perm = rng.permutation(n_r)
        ranks, ranks_null, signs, signs_null = [], [], [], []
        for pi, col in enumerate(cols):
            col = int(col)
            null_col = int(perm[col])
            delta = raw[pi, hi_i] - raw[pi, lo_i]
            mag = np.abs(delta)
            ranks.append(int((mag > mag[col]).sum()))
            ranks_null.append(int((mag > mag[null_col]).sum()))
            signs.append(float(delta[col]) > 0)
            signs_null.append(float(delta[null_col]) > 0)
        ranks, ranks_null = np.asarray(ranks), np.asarray(ranks_null)
        signs, signs_null = np.asarray(signs), np.asarray(signs_null)
        rank_frac = float(np.median(ranks)) / max(n_r - 1, 1)
        rank_frac_null = float(np.median(ranks_null)) / max(n_r - 1, 1)
        sign_rate, sign_rate_null = float(signs.mean()), float(signs_null.mean())
        own_largest = float((ranks == 0).mean())
        separated = {
            "own_largest": own_largest >= C.SELF_RESPONSE_MIN_PASS,
            "self_rank": rank_frac <= ALIGN_RANK_MAX_FRACTION
            and rank_frac_null >= ALIGN_RANK_NULL_MIN_FRACTION,
            "sign": sign_rate >= ALIGN_SIGN_MIN and sign_rate_null <= ALIGN_SIGN_NULL_MAX,
        }
        rows.append(
            {
                "anchor": anchor,
                "n_perturbed": int(n_p),
                "n_readout": int(n_r),
                "own_largest": own_largest,
                "own_largest_null": float((ranks_null == 0).mean()),
                "self_rank_median": float(np.median(ranks)),
                "self_rank_median_fraction": rank_frac,
                "self_rank_median_null": float(np.median(ranks_null)),
                "self_rank_median_fraction_null": rank_frac_null,
                "sign_rate": sign_rate,
                "sign_rate_null": sign_rate_null,
                "separated_by": sorted(k for k, v in separated.items() if v),
                "aligned": any(separated.values()),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# aggregate: runs on the pod
# --------------------------------------------------------------------------- #
def verify_all(bundles_dir: Path, ch: Checks) -> list[dict]:
    bundles = [json.loads(p.read_text()) for p in sorted(bundles_dir.glob("report_bundle*.json"))]
    have = {(b["model"], b["cell_line"]) for b in bundles}
    want = {(m, c) for m in C.MODELS for c in C.CELL_LINES}
    ch.add(
        "coverage",
        "model_x_cell_line",
        have == want,
        f"{len(have)}/{len(want)} blocks present; missing {sorted(want - have) or 'none'}",
    )

    # every block has to say how much of the in-scope perturbation set it covers, and
    # the mix across models has to be stated rather than discovered later
    missing_cov = [
        f"{b['model']}/{b['cell_line']}" for b in bundles if not b["manifest"].get("coverage")
    ]
    mix = sorted(
        {
            (
                b["manifest"].get("coverage", {}).get("mode", "unrecorded"),
                b["manifest"].get("coverage", {}).get("n_perturbed"),
                b["manifest"].get("coverage", {}).get("n_in_scope"),
            )
            for b in bundles
        }
    )
    ch.add(
        "coverage",
        "perturbed_gene_coverage_recorded",
        not missing_cov,
        "coverage present in every block manifest; the tree mixes "
        + "; ".join(f"{m} at {n} of {a}" for m, n, a in mix)
        + (f"; missing in {missing_cov}" if missing_cov else ""),
    )
    for b in bundles:
        cov = b["manifest"].get("coverage", {})
        if cov.get("mode") != "full":
            continue
        anchors = b["manifest"].get("anchors", {})
        full_anchors = [
            a for a, meta in anchors.items() if meta.get("n_perturbed") == cov["n_perturbed"]
        ]
        ch.add(
            "coverage",
            f"{b['model']}/{b['cell_line']}/full_at_every_anchor",
            len(full_anchors) == len(anchors),
            f"{len(full_anchors)} of {len(anchors)} anchors run the whole "
            f"{cov['n_perturbed']}-gene in-scope set"
            + (
                f"; masked variant at {b['manifest'].get('masked', {}).get('n_perturbed')}"
                if b["manifest"].get("masked", {}).get("ran")
                else ""
            ),
        )

    for b in bundles:
        sec = f"{b['model']}/{b['cell_line']}"
        anchors = {k.split("/")[0] for k in b["blocks"]}
        expected = set(C.ANCHORS)
        if b["model"] == "transcriptformer":
            pass
        ch.add(
            "coverage",
            f"{sec}/anchors",
            expected <= anchors,
            f"anchors present: {sorted(anchors)}",
        )
        forms = {k.split("/")[1] for k in b["blocks"] if not k.split("/")[0].endswith("_masked")}
        ch.add(
            "coverage",
            f"{sec}/forms",
            set(FORMS) <= forms,
            f"forms present: {sorted(forms)} plus the gene-embedding table",
        )
        align = b.get("alignment", [])
        if align:
            ch.add(
                "alignment",
                sec,
                all(r["aligned"] for r in align),
                "; ".join(
                    f"{r['anchor']} on {'+'.join(r['separated_by']) or 'no statistic'}"
                    for r in align
                ),
            )
        failed = [r for r in b.get("checks", []) if not r["ok"]]
        ch.add(
            "block_checks",
            sec,
            not failed,
            f"{len(b.get('checks', []))} block-level checks, {len(failed)} failed"
            + (f": {[r['check'] for r in failed][:6]}" if failed else ""),
        )

    # the visible / masked / median-substituted comparison, where it applies
    hidden_dir = paths.repo_dir() / "results" / "report" / "hidden_readout"
    if hidden_dir.exists():
        bidirectional = sorted(
            {b["model"] for b in bundles if b["manifest"].get("masked", {}).get("ran")}
        )
        for model in bidirectional:
            for cl in C.CELL_LINES:
                path = hidden_dir / f"{model}_{cl}.json"
                vpath = hidden_dir / f"verification_{model}_{cl}.json"
                if not path.exists():
                    ch.add(
                        "hidden_readout",
                        f"{model}/{cl}",
                        False,
                        "no visible / masked / median-substituted comparison for a model whose "
                        "masked variant ran",
                    )
                    continue
                rec = json.loads(path.read_text())
                checks = json.loads(vpath.read_text()) if vpath.exists() else {"checks": []}
                own = checks.get("checks", [])
                failed = [r["check"] for r in own if not r["ok"]]
                ratios = rec.get("ratios", {})
                ch.add(
                    "hidden_readout",
                    f"{model}/{cl}",
                    not failed and bool(ratios),
                    f"{rec['n_perturbed']} perturbed genes, substitute "
                    f"{rec['substitute_level']['value_cp10k']:.4g} CP10K; "
                    + "; ".join(
                        f"{k} {v:.4g}" for k, v in ratios.items() if isinstance(v, (int, float))
                    )
                    + f"; {sum(1 for r in own if r['ok'])}/{len(own)} own checks passed"
                    + (f", failed {failed}" if failed else ""),
                )

    # the floor sweep and the read-out set it selected
    floor = paths.repo_dir() / "results" / "floor_sweep"
    decision = json.loads((floor / "floor_decision.json").read_text())
    summary = pd.read_parquet(floor / "floor_ceiling_summary.parquet")
    ran = {(r.cell_line, r.split, r.floor_cp10k) for r in summary.itertuples()}
    want_ran = {
        (cl, sp, fl)
        for cl in C.CELL_LINES
        for sp in ("split_half_cells", "split_by_batch")
        for fl in C.FLOOR_GRID_CP10K
    }
    ch.add(
        "floor_sweep",
        "grid_complete",
        want_ran <= ran,
        f"{len(ran)} (cell line, split, floor) cells computed over floors "
        f"{list(C.FLOOR_GRID_CP10K)}",
    )
    for cl in C.CELL_LINES:
        d = decision["decisions"][cl]
        readout = pd.read_parquet(
            paths.handoff_root() / "readout" / f"readout_genes_{cl}.parquet"
        )
        floor_used = float(d["chosen_floor_cp10k"])
        ok = bool((readout.pooled_control_cp10k >= floor_used).all()) and readout.shape[0] == d[
            "n_readout_genes"
        ]
        ch.add(
            "floor_sweep",
            f"{cl}/readout_matches_floor",
            ok,
            f"{readout.shape[0]} read-out genes, all at or above the selected "
            f"{floor_used} CP10K floor (plateau found: {d['plateau_found']})",
        )
        shared = decision["shared_response"]["by_cell_line_stratum_floor"]
        keys = [k for k in shared if k.startswith(f"{cl}|")]
        ch.add(
            "floor_sweep",
            f"{cl}/shared_response_recorded",
            len(keys) == len(C.PERTURBED_STRATA) * len(C.FLOOR_GRID_CP10K),
            f"{len(keys)} (stratum, floor) cells carry ceiling, mean-effect baseline, "
            "cross-perturbation correlation and PC1 share",
        )
    return bundles


def write_summaries(bundles_dir: Path, out: Path) -> None:
    """Two small tables carrying the numbers the report and the checkpoints cite.

    ``block_summary`` is one row per (model, cell line, anchor): the selected
    step and its verdict beside the three alignment statistics and their nulls.
    ``step_stability`` is the adjacent-step agreement curve behind that verdict.
    """
    block_rows, step_rows = [], []
    for p in sorted(bundles_dir.glob("report_bundle*.json")):
        b = json.loads(p.read_text())
        model, cl = b["model"], b["cell_line"]
        align = {r["anchor"]: r for r in b.get("alignment", [])}
        for anchor, sel in b["manifest"].get("step_selection", {}).items():
            a = align.get(anchor, {})
            block_rows.append(
                {
                    "model": model,
                    "cell_line": cl,
                    "anchor": anchor,
                    "n_perturbed": a.get("n_perturbed"),
                    "n_readout": b["n_readout"],
                    "selected_step": sel["selected_step"],
                    "step_verdict": sel["verdict"],
                    "gradient_usable_for": sel["gradient_features_usable_for"],
                    "own_largest": a.get("own_largest"),
                    "own_largest_null": a.get("own_largest_null"),
                    "self_rank_median": a.get("self_rank_median"),
                    "self_rank_median_null": a.get("self_rank_median_null"),
                    "sign_rate": a.get("sign_rate"),
                    "sign_rate_null": a.get("sign_rate_null"),
                    "separated_by": "+".join(a.get("separated_by", [])) or "none",
                    "aligned": a.get("aligned"),
                }
            )
        for r in b.get("step_sweep", []):
            step_rows.append({"model": model, "cell_line": cl, **r})
    (out / "block_summary.json").write_text(json.dumps(block_rows, indent=2, default=str))
    (out / "step_stability.json").write_text(json.dumps(step_rows, indent=2, default=str))
    if block_rows:
        pd.DataFrame(block_rows).to_parquet(out / "block_summary.parquet", index=False)
    if step_rows:
        pd.DataFrame(step_rows).to_parquet(out / "step_stability.parquet", index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block-dir", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--bundles", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ch = Checks()
    if args.block_dir == "auto":
        import os

        mount = Path(os.environ.get("SILICO_REUSED_VOLUME_DIR", ""))
        hits = sorted(mount.glob("**/features/*/*/manifest.json"))
        if len(hits) != 1:
            raise SystemExit(f"expected one feature block in {mount}, found {len(hits)}: {hits[:4]}")
        args.block_dir = str(hits[0].parent)
        print(f"[verify] block {args.block_dir}", flush=True)
    if args.block_dir:
        block = Path(args.block_dir)
        bundle = verify_block(block, ch)
        out = Path(args.out) if args.out else block
        paths.ensure(out)
        (out / "verification.json").write_text(json.dumps({"checks": ch.rows}, indent=2))
        name = f"report_bundle_{bundle['model']}_{bundle['cell_line']}.json"
        (out / name).write_text(json.dumps(bundle, indent=2, default=str))
        print(f"wrote {out / name}", flush=True)
    if args.all:
        bundles_dir = Path(args.bundles) if args.bundles else paths.report_dir() / "bundles"
        verify_all(bundles_dir, ch)

    ok, total = ch.summary()
    print(f"\n{ok}/{total} checks passed", flush=True)
    if args.all:
        out = paths.ensure(paths.repo_dir() / "results" / "verification")
        write_summaries(Path(args.bundles) if args.bundles else paths.report_dir() / "bundles", out)
        (out / "verify_all.json").write_text(json.dumps({"checks": ch.rows}, indent=2))
        lines = [f"[{'PASS' if r['ok'] else 'FAIL'}] {r['section']}/{r['check']}: {r['detail']}" for r in ch.rows]
        (out / "verify_all.txt").write_text("\n".join(lines) + f"\n\n{ok}/{total} checks passed\n")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
