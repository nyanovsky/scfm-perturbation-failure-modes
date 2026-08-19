"""Extract one model's response features for one cell line, at three anchors.

One job per (model, cell line). Everything a later run needs to interpret a
block is written next to it, because a feature block on its own is a matrix of
numbers whose meaning depends entirely on the anchor, the value grid, the
finite-difference step, the supplied gene set and the model's output scale.

What "response" and "response difference" mean here
---------------------------------------------------
The response at read-out gene *j* is the single scalar the model emits for that
gene given a whole-cell input vector. A response difference is that scalar from
two forward passes identical except for the input value of the perturbed gene
*i*:

    e_ij = M(x_{i <- p})_j - M(x_{i <- b})_j

with *b* the anchor's own value for gene *i* and *p* the perturbed value. Sign
convention throughout: perturbed minus baseline, so a positive entry means
raising gene *i* raises the model's prediction for gene *j*. Zero-out is the
special case p = 0, reported with that same convention rather than the paper's
reversed one.

The four forms
--------------
``zero_out``  one value per pair: the response difference at p = 0.
``vvp``       Virtual Value Perturbation: the response difference at each value
              of the grid, concatenated.
``gdt``       Gradient Trajectory: the derivative of the read-out gene's output
              with respect to the perturbed gene's input, at each base value of
              the same grid, concatenated. Exact forward-mode where the model
              supports it, central difference at a selected step otherwise.
``gene_embedding``  each model's own vocabulary embedding for the read-out genes,
              the pair feature the comparison run uses as its no-response
              baseline.

One forward pass at each grid value serves both VVP (the value itself) and GDT
(the centre of its difference), which is why the two grids are the same grid.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import paths
from ..models import queries as Q
from ..models.base import environment_manifest
from ..forms import normalize_forms
from .scope import coverage_record, select_perturbed

#: native-input floor: none of the four models takes a negative input value
NATIVE_FLOOR = 0.0


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load_inputs(model: str, cell_line: str) -> dict:
    universe = pd.read_parquet(paths.universe_dir() / "gene_universe.parquet")
    vocab = pd.read_parquet(paths.universe_dir() / f"vocabulary_{model}.parquet")
    readout = pd.read_parquet(
        paths.handoff_root() / "readout" / f"readout_genes_{cell_line}.parquet"
    )
    perts = pd.read_parquet(paths.effects_dir(cell_line) / "perturbations.parquet")
    demo = paths.demo_cells_dir() / f"demo_cells_{cell_line}.parquet"
    cells = pd.read_parquet(demo) if demo.exists() else None
    return {
        "universe": universe,
        "vocab": vocab,
        "readout": readout,
        "perturbations": perts,
        "cells": cells,
        "cells_path": str(demo) if demo.exists() else None,
    }


def build_anchors(readout: pd.DataFrame, cells: pd.DataFrame | None) -> dict:
    """The three query anchors, each a CP10K vector over the read-out genes."""
    n = readout.shape[0]
    control = readout["pooled_control_cp10k"].to_numpy(dtype=np.float64)
    uniform_level = C.UNIFORM_ANCHOR_TOTAL_CP10K / n
    out = {
        "uniform_virtual": {
            "profiles": [np.full(n, uniform_level, dtype=np.float64)],
            "description": (
                f"every supplied gene pinned at {uniform_level:.4f} CP10K, the uniform "
                "share of a 10,000-count virtual cell; the paper's construction, and far "
                "off the distribution the models were trained on"
            ),
            "labels": ["uniform"],
        },
        "mean_control": {
            "profiles": [control],
            "description": (
                "the pooled non-targeting control mean expression profile of this cell "
                "line, restricted to the read-out genes; the on-distribution counterpart"
            ),
            "labels": ["mean_control"],
        },
    }
    if cells is not None:
        cols = [c for c in cells.columns if c.startswith("counts__")][: C.N_INDIVIDUAL_CELLS]
        gene_ids = cells["gene_id"].to_numpy().astype(str)
        want = readout["gene_id"].to_numpy().astype(str)
        pos = {g: i for i, g in enumerate(gene_ids)}
        take = np.array([pos.get(g, -1) for g in want])
        if (take < 0).any():
            raise RuntimeError("a read-out gene is missing from the demonstration cells")
        profiles, labels = [], []
        for col in cols:
            counts = cells[col].to_numpy(dtype=np.float64)
            total = float(counts.sum())
            profiles.append(counts[take] / max(total, 1.0) * 1e4)
            labels.append(col.removeprefix("counts__"))
        out["individual_cells"] = {
            "profiles": profiles,
            "description": (
                f"{len(profiles)} real non-targeting control cells, each converted to "
                "CP10K so every anchor sits at one depth; responses are averaged over "
                "the cells and their spread is recorded"
            ),
            "labels": labels,
        }
    return out


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
def query_batch(
    model, x0: np.ndarray, col: int, values_native: np.ndarray, step: float
) -> np.ndarray:
    """One perturbed gene's whole grid: anchor row, then (v-h, v, v+h) per value."""
    rows = [x0]
    for v in values_native:
        lo = max(float(v) - step, NATIVE_FLOOR)
        for w in (lo, float(v), float(v) + step):
            x = x0.copy()
            x[col] = w
            rows.append(x)
    return np.stack(rows)


def extract_anchor(
    model,
    anchor_cp10k_profiles: list[np.ndarray],
    perturbed: pd.DataFrame,
    step: float,
    grid_cp10k: tuple[float, ...],
    log_every: int = 25,
    exact_derivative: bool = True,
) -> dict:
    """Run every perturbed gene at one anchor and assemble the four forms."""
    cols = perturbed["readout_col"].to_numpy(dtype=np.int64)
    n_p, n_r, n_v = cols.size, model.n_readout, len(grid_cp10k)
    values_native = np.asarray(model.to_native(np.asarray(grid_cp10k, dtype=np.float64)))

    acc = {
        "vvp": np.zeros((n_p, n_v, n_r), dtype=np.float64),
        "gdt": np.zeros((n_p, n_v, n_r), dtype=np.float64),
        "y_pert": np.zeros((n_p, n_v, n_r), dtype=np.float64),
        "y_base": np.zeros((n_p, n_v, n_r), dtype=np.float64),
        "gdt_exact": np.zeros((n_p, n_v, n_r), dtype=np.float64),
    }
    have_exact = np.zeros(n_p, dtype=bool)
    rows: list[dict] = []
    started = time.time()

    for ci, profile in enumerate(anchor_cp10k_profiles):
        x0 = np.asarray(model.to_native(profile), dtype=np.float32)
        for pi, col in enumerate(cols):
            batch = query_batch(model, x0, int(col), values_native, step)
            resp = model.forward_batch(batch, perturbed_col=int(col)).astype(np.float64)
            base = resp[0]
            for vi in range(n_v):
                lo, mid, hi = resp[1 + 3 * vi], resp[2 + 3 * vi], resp[3 + 3 * vi]
                v = float(values_native[vi])
                lo_v = max(v - step, NATIVE_FLOOR)
                acc["vvp"][pi, vi] += mid - base
                acc["gdt"][pi, vi] += (hi - lo) / max(float(v) + step - lo_v, 1e-12)
                acc["y_pert"][pi, vi] += mid
                acc["y_base"][pi, vi] += base
                d = mid - base
                rows.append(
                    {
                        "cell_index": ci,
                        "perturbed_gene": str(perturbed.gene_id.iloc[pi]),
                        "target": str(perturbed.target.iloc[pi]),
                        "value_cp10k": float(grid_cp10k[vi]),
                        "value_native": v,
                        "cross_gene_mean": float(d.mean()),
                        "cross_gene_sd": float(d.std()),
                        "max_abs": float(np.abs(d).max()),
                        "self_delta": float(d[int(col)]),
                        "self_rank_by_abs": int((np.abs(d) > abs(d[int(col)])).sum()),
                        "n_nonfinite": int((~np.isfinite(d)).sum()),
                        "baseline_response_median": float(np.median(base)),
                    }
                )
            if exact_derivative:
                for vi in range(n_v):
                    x = x0.copy()
                    x[int(col)] = float(values_native[vi])
                    jvp = model.exact_jvp(x, int(col))
                    if jvp is None:
                        exact_derivative = False
                        break
                    acc["gdt_exact"][pi, vi] += jvp
                    have_exact[pi] = True
            if log_every and (pi + 1) % log_every == 0:
                done = ci * n_p + pi + 1
                rate = (time.time() - started) / done
                print(
                    f"    cell {ci} gene {pi + 1}/{n_p} ({rate:.2f}s per perturbed gene, "
                    f"{rate * (len(anchor_cp10k_profiles) * n_p - done) / 60:.1f} min left)",
                    flush=True,
                )

    k = float(len(anchor_cp10k_profiles))
    for key in acc:
        acc[key] /= k
    zero_index = list(grid_cp10k).index(0.0) if 0.0 in grid_cp10k else 0
    forms = {
        "vvp": normalize_forms(acc["vvp"], acc["y_pert"], acc["y_base"], model),
        "zero_out": normalize_forms(
            acc["vvp"][:, zero_index], acc["y_pert"][:, zero_index], acc["y_base"][:, zero_index], model
        ),
    }
    gdt_primary = acc["gdt_exact"] if have_exact.all() else acc["gdt"]
    forms["gdt"] = {
        k2: v2
        for k2, v2 in normalize_forms(gdt_primary, gdt_primary, gdt_primary, model).items()
        if k2 != "log2fc"  # a derivative has no fold change
    }
    diagnostics = {
        "n_cells_averaged": len(anchor_cp10k_profiles),
        "zero_out_value_index": zero_index,
        "gdt_source": "forward_mode_autodiff_jvp" if have_exact.all() else
        f"central_finite_difference(step={step:.4g})",
        # measured, not assumed: whether this model's production layout admits a
        # forward-mode derivative is decided by trying it here, and the reason it
        # did not is recorded verbatim rather than left to a contract default
        "forward_mode": {
            "attempted": True,
            "available": bool(have_exact.all()),
            "reason_unavailable": None if have_exact.all() else str(
                getattr(model, "_jvp_error", "the adapter returned no forward-mode column")
            ),
        },
        "seconds": round(time.time() - started, 1),
    }
    if have_exact.any():
        # per value, because the estimator is not the same at every value: at 0 the
        # central difference has nowhere to go below the count floor, so it becomes a
        # one-sided difference and disagrees with the exact derivative wherever the
        # response is curved there. Pooling the values hides that in one number.
        per_value = []
        for vi in range(n_v):
            a = acc["gdt_exact"][:, vi].ravel()
            b = acc["gdt"][:, vi].ravel()
            ok = np.isfinite(a) & np.isfinite(b)
            denom = np.linalg.norm(a[ok]) + np.linalg.norm(b[ok])
            per_value.append(
                {
                    "value_cp10k": float(grid_cp10k[vi]),
                    "one_sided": bool(float(values_native[vi]) - step < NATIVE_FLOOR),
                    "pearson": float(np.corrcoef(a[ok], b[ok])[0, 1])
                    if ok.sum() > 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0
                    else None,
                    "relative_l2": float(2 * np.linalg.norm(a[ok] - b[ok]) / denom)
                    if denom
                    else None,
                }
            )
        two_sided = [r for r in per_value if not r["one_sided"] and r["pearson"] is not None]
        diagnostics["exact_vs_finite_difference"] = {
            "per_value": per_value,
            "median_pearson_two_sided": float(np.median([r["pearson"] for r in two_sided]))
            if two_sided
            else None,
            "median_relative_l2_two_sided": float(
                np.median([r["relative_l2"] for r in two_sided])
            )
            if two_sided
            else None,
            "step": step,
            "note": "the exact forward-mode derivative is the feature; the central "
            "difference at the selected step is the cross-check. A value whose "
            "difference would cross the input floor is marked one_sided and excluded "
            "from the medians",
        }
    if len(anchor_cp10k_profiles) > 1:
        diagnostics["per_cell_spread"] = "responses averaged over cells; see summary rows"
    return {"forms": forms, "summary": pd.DataFrame(rows), "diagnostics": diagnostics}


# --------------------------------------------------------------------------- #
# step-size stability
# --------------------------------------------------------------------------- #
def step_sweep(model, anchor_cp10k: np.ndarray, cols: np.ndarray, base_value_cp10k: float) -> tuple[pd.DataFrame, dict]:
    """Central-difference columns at a range of steps, and the selected step."""
    steps = tuple(C.STEP_GRID[model.name])
    x0 = np.asarray(model.to_native(anchor_cp10k), dtype=np.float32)
    v = float(np.asarray(model.to_native(np.array([base_value_cp10k])))[0])
    columns: dict[float, np.ndarray] = {}
    for step in steps:
        batch, denom = [], []
        for col in cols:
            lo = max(v - step, NATIVE_FLOOR)
            for w in (lo, v + step):
                x = x0.copy()
                x[int(col)] = w
                batch.append(x)
            denom.append(v + step - lo)
        resp = model.forward_batch(np.stack(batch), perturbed_col=int(cols[0])).astype(np.float64)
        cols_out = []
        for i in range(cols.size):
            cols_out.append((resp[2 * i + 1] - resp[2 * i]) / max(denom[i], 1e-12))
        columns[step] = np.stack(cols_out)
        print(f"    step {step}: |column| median {np.median(np.abs(columns[step])):.4g}", flush=True)

    rows = []
    for a, b in zip(steps[:-1], steps[1:]):
        ca, cb = columns[a], columns[b]
        pear, rel = [], []
        for i in range(cols.size):
            x, y = ca[i], cb[i]
            if np.std(x) > 0 and np.std(y) > 0:
                pear.append(float(np.corrcoef(x, y)[0, 1]))
            denom = np.linalg.norm(x) + np.linalg.norm(y)
            if denom > 0:
                rel.append(float(2 * np.linalg.norm(x - y) / denom))
        rows.append(
            {
                "step": a,
                "next_step": b,
                "median_pearson_adjacent": float(np.median(pear)) if pear else float("nan"),
                "median_relative_l2_adjacent": float(np.median(rel)) if rel else float("nan"),
                "median_abs_value": float(np.median(np.abs(ca))),
                "n_genes": int(cols.size),
            }
        )
    table = pd.DataFrame(rows)
    stable = table[
        (table.median_pearson_adjacent >= C.STEP_STABLE_PEARSON)
        & (table.median_relative_l2_adjacent <= C.STEP_STABLE_RELATIVE_L2)
    ]
    if not stable.empty:
        chosen = float(stable.iloc[0].step)
        verdict = "stable_in_direction_and_magnitude"
        usable = "direction, ranking and magnitude"
    else:
        best = table.iloc[int(np.nanargmax(table.median_pearson_adjacent.to_numpy()))]
        chosen = float(best.step)
        verdict = "magnitude_unstable"
        usable = "direction and ranking only; magnitudes move with the step"
    selection = {
        "rule": (
            f"smallest step whose column agrees with the next larger step at median "
            f"Pearson >= {C.STEP_STABLE_PEARSON} and median relative L2 difference <= "
            f"{C.STEP_STABLE_RELATIVE_L2}; if none qualifies the step with the best "
            "direction agreement is used and the block is flagged"
        ),
        "steps_swept": list(steps),
        "base_value_cp10k": base_value_cp10k,
        "n_genes": int(cols.size),
        "selected_step": chosen,
        "verdict": verdict,
        "gradient_features_usable_for": usable,
        "adjacent_agreement": table.to_dict(orient="records"),
    }
    return table, selection


# --------------------------------------------------------------------------- #
# wiring checks
# --------------------------------------------------------------------------- #
def wiring_checks(vvp: np.ndarray, cols: np.ndarray, grid: tuple[float, ...], seed: int = C.SEED) -> dict:
    """Would a gene-index misalignment show up?

    Two readings of the same numbers. The **sign** check is the plan's: raising a
    perturbed gene's input should raise that gene's own predicted output, read
    between the lowest positive value on the grid and the highest. The
    **magnitude** check is the one that actually guards the wiring: the perturbed
    gene's own entry should be the largest response in the vector, whatever its
    sign. A permutation of the read-out order must destroy the magnitude check,
    which is what shows the guard has power.
    """
    positive = [i for i, v in enumerate(grid) if v > 0]
    lo_i, hi_i = positive[0], positive[-1]
    n_p, n_r = vvp.shape[0], vvp.shape[2]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_r)

    sign_pass, rank_pass, perm_pass, ranks = 0, 0, 0, []
    for pi in range(n_p):
        col = int(cols[pi])
        delta_hi, delta_lo = vvp[pi, hi_i], vvp[pi, lo_i]
        slope = float(delta_hi[col] - delta_lo[col])
        sign_pass += int(slope > 0)
        mag = np.abs(delta_hi - delta_lo)
        rank = int((mag > mag[col]).sum())
        ranks.append(rank)
        rank_pass += int(rank == 0)
        perm_pass += int((mag > mag[perm[col]]).sum() == 0)
    return {
        "n_genes": int(n_p),
        "value_low_cp10k": float(grid[lo_i]),
        "value_high_cp10k": float(grid[hi_i]),
        "sign_check_pass_fraction": sign_pass / max(n_p, 1),
        "sign_check_note": "fraction whose own predicted output RISES when its input "
        "rises; a model that reverts to a population level can fail this while still "
        "being correctly wired",
        "magnitude_check_pass_fraction": rank_pass / max(n_p, 1),
        "magnitude_check_note": "fraction whose own entry is the largest response in "
        "the vector; this is the wiring guard",
        "median_self_rank": float(np.median(ranks)),
        "permuted_magnitude_pass_fraction": perm_pass / max(n_p, 1),
        "permutation_note": "the same test after shuffling the read-out gene order; it "
        "must collapse to chance for the guard to have power",
        "threshold": C.SELF_RESPONSE_MIN_PASS,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(
    model_name: str,
    cell_line: str,
    cache: Path,
    out_dir: Path,
    scope: C.Scope,
    device: str = "cuda",
    anchors: tuple[str, ...] = C.ANCHORS,
    batch: int | None = None,
    supplied_mode: str | None = None,
    image: str = "unrecorded",
    do_masked: bool = True,
    do_context_compare: bool = True,
) -> dict:
    started = time.time()
    data = load_inputs(model_name, cell_line)
    readout = data["readout"]
    perturbed = select_perturbed(data["perturbations"], readout, scope)
    coverage = coverage_record(data["perturbations"], readout, scope, perturbed)
    print(
        f"[{model_name}/{cell_line}] {readout.shape[0]} read-out genes, "
        f"{perturbed.shape[0]} perturbed genes of {coverage['n_in_scope']} in scope "
        f"(coverage {coverage['mode']}, fraction {coverage['fraction_of_scope']:.3f}) "
        f"({perturbed.effect_stratum.value_counts().to_dict()})",
        flush=True,
    )

    model = Q.build(model_name, cache, data["universe"], data["vocab"], device=device)
    model.load()
    if batch:
        model.batch_size = int(batch)
    anchor_defs = build_anchors(readout, data["cells"])
    control = readout["pooled_control_cp10k"].to_numpy(dtype=np.float64)
    kwargs = (
        {"supplied_mode": supplied_mode or C.SCGPT_PRODUCTION_MODE}
        if model_name == "scgpt"
        else {}
    )
    layout = model.configure(readout, control, cell_line, **kwargs)
    print(f"[{model_name}/{cell_line}] layout {json.dumps(layout.as_dict(), default=str)}", flush=True)

    out_dir = paths.ensure(out_dir)
    readout.to_parquet(out_dir / "readout_genes.parquet", index=False)
    perturbed.to_parquet(out_dir / "perturbed_genes.parquet", index=False)
    emb, emb_source = model.gene_embeddings()
    np.savez(
        out_dir / "gene_embeddings.npz",
        embedding=emb.astype(np.float32),
        gene_id=readout["gene_id"].to_numpy().astype(str),
    )

    manifest: dict = {
        "model": model_name,
        "cell_line": cell_line,
        "layout": layout.as_dict(),
        "scope": {
            "n_perturbed": int(perturbed.shape[0]),
            "n_perturbed_cells_anchor": int(perturbed.in_cells_anchor.sum()),
            "n_perturbed_masked": int(perturbed.in_masked.sum()),
            "n_step_sweep": int(perturbed.in_step_sweep.sum()),
            "strata": list(scope.strata),
        },
        "coverage": coverage,
        "value_grid_cp10k": list(C.VALUE_GRID_CP10K),
        "gdt_base_grid_cp10k": list(C.GDT_BASE_GRID_CP10K),
        "sign_convention": "perturbed minus baseline; a positive entry means raising the "
        "perturbed gene raises the model's prediction for the read-out gene. Zero-out "
        "uses this convention, not the paper's reversed one",
        "expression_floor_cp10k": C.RUN1_FLOOR_CP10K,
        "expression_floor_note": "the read-out set is the inclusive floor; each gene's "
        "pooled control level travels with the block so a higher floor is a column mask",
        "gene_embedding_source": emb_source,
        "gene_embedding_shape": list(emb.shape),
        "mask_spec": model.mask_spec(),
        "seed": C.SEED,
        "anchors": {},
        "substitutions": list(getattr(model, "substitutions", [])),
        "contract": json.loads(model.contract().to_json()) if hasattr(model, "contract") else {},
    }

    # ---- step stability, per anchor, before any gradient feature is written
    step_rows, steps_by_anchor = [], {}
    sweep_cols = perturbed.loc[perturbed.in_step_sweep, "readout_col"].to_numpy(dtype=np.int64)
    for anchor in anchors:
        if anchor not in anchor_defs:
            continue
        print(f"[{model_name}/{cell_line}] step sweep at {anchor}", flush=True)
        table, selection = step_sweep(
            model, anchor_defs[anchor]["profiles"][0], sweep_cols, base_value_cp10k=4.0
        )
        table["anchor"] = anchor
        step_rows.append(table)
        steps_by_anchor[anchor] = selection
        print(
            f"[{model_name}/{cell_line}] {anchor}: step {selection['selected_step']} "
            f"({selection['verdict']})",
            flush=True,
        )
    pd.concat(step_rows, ignore_index=True).to_parquet(out_dir / "step_sweep.parquet", index=False)
    (out_dir / "step_selection.json").write_text(json.dumps(steps_by_anchor, indent=2))
    manifest["step_selection"] = steps_by_anchor

    # ---- the three anchors
    summaries = []
    for anchor in anchors:
        if anchor not in anchor_defs:
            print(f"[{model_name}/{cell_line}] anchor {anchor} unavailable, skipped", flush=True)
            continue
        spec = anchor_defs[anchor]
        subset = perturbed[perturbed.in_cells_anchor] if anchor == "individual_cells" else perturbed
        step = float(steps_by_anchor[anchor]["selected_step"])
        print(
            f"[{model_name}/{cell_line}] anchor {anchor}: {subset.shape[0]} genes x "
            f"{len(spec['profiles'])} profile(s), step {step}",
            flush=True,
        )
        result = extract_anchor(
            model,
            spec["profiles"],
            subset,
            step,
            C.VALUE_GRID_CP10K,
            exact_derivative=True,
        )
        adir = paths.ensure(out_dir / f"anchor={anchor}")
        for form, arrays in result["forms"].items():
            np.savez(
                adir / f"{form}.npz",
                **arrays,
                perturbed_gene_id=subset.gene_id.to_numpy().astype(str),
                readout_gene_id=readout["gene_id"].to_numpy().astype(str),
                value_grid_cp10k=np.asarray(C.VALUE_GRID_CP10K, dtype=np.float64),
            )
        result["summary"]["anchor"] = anchor
        result["summary"].to_parquet(adir / "query_summary.parquet", index=False)
        summaries.append(result["summary"])
        manifest["anchors"][anchor] = {
            "description": spec["description"],
            "cells": spec["labels"],
            "n_perturbed": int(subset.shape[0]),
            "step": step,
            "forms": {f: {k: list(v.shape) for k, v in a.items()} for f, a in result["forms"].items()},
            **result["diagnostics"],
        }
        manifest.setdefault("wiring_checks", {})[anchor] = wiring_checks(
            result["forms"]["vvp"]["raw"],
            subset["readout_col"].to_numpy(dtype=np.int64),
            C.VALUE_GRID_CP10K,
        )
        print(
            f"[{model_name}/{cell_line}] wiring {anchor} "
            f"{json.dumps(manifest['wiring_checks'][anchor])}",
            flush=True,
        )

    # ---- masked variant of the real-background anchor
    if do_masked and model.mask_spec() is not None:
        manifest["masked"] = run_masked(
            model,
            anchor_defs,
            perturbed,
            out_dir,
            steps_by_anchor,
            readout["gene_id"].to_numpy().astype(str),
        )
    else:
        manifest["masked"] = {
            "ran": False,
            "reason": "the model is autoregressive: a gene never sees its own value, so it "
            "is natively in the masked regime and a masked variant would be the same query",
        }

    # ---- scGPT context width comparison and its untiled exact-derivative check
    if do_context_compare and model_name == "scgpt":
        manifest["context_comparison"] = compare_context_widths(
            model, data, readout, perturbed, out_dir
        )
        manifest["untiled_exact_check"] = untiled_exact_check(
            model, readout, perturbed, float(steps_by_anchor["mean_control"]["selected_step"])
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    # the report reads the perturbed genes' own anchor levels to interpret the
    # individual-cell anchor, and the pod cannot read an npz back out of the store,
    # so the small index table travels separately, tagged by coverage
    blocks_dir = paths.ensure(paths.report_dir() / "blocks")
    tag = "full" if coverage["mode"] == "full" else f"n{coverage['n_perturbed']}"
    perturbed.to_parquet(blocks_dir / f"perturbed_genes_{tag}_{cell_line}.parquet", index=False)
    (out_dir / "environment.json").write_text(
        json.dumps(
            environment_manifest(model_name, image, {"device": device, "cell_line": cell_line}),
            indent=2,
            default=str,
        )
    )
    manifest["seconds"] = round(time.time() - started, 1)
    print(f"[{model_name}/{cell_line}] done in {manifest['seconds']}s", flush=True)
    return manifest


def run_masked(
    model,
    anchor_defs: dict,
    perturbed: pd.DataFrame,
    out_dir: Path,
    steps: dict,
    readout_ids: np.ndarray,
) -> dict:
    """The masked counterpart of the mean-control anchor.

    Three of these models are bidirectional and see the read-out gene's own input
    value while predicting that gene's output, so an unmasked response measures
    only what survives after the model has been told the answer. Masking the
    read-out gene removes that, and comparing the two at the SAME anchor is what
    separates the masking regime from the background it sits on.
    """
    spec = model.mask_spec()
    rate = float(spec["rate"])
    blocks = Q.partition_masks(model.n_readout, rate, C.MASK_MIN_COVERAGE, C.SEED)
    subset = perturbed[perturbed.in_masked]
    cols = subset["readout_col"].to_numpy(dtype=np.int64)
    grid = C.MASKED_VALUE_GRID_CP10K
    values_native = np.asarray(model.to_native(np.asarray(grid, dtype=np.float64)))
    x0 = np.asarray(model.to_native(anchor_defs["mean_control"]["profiles"][0]), dtype=np.float32)
    started = time.time()

    coverage = np.zeros(model.n_readout, dtype=np.int64)
    base_acc = np.zeros(model.n_readout, dtype=np.float64)
    for block in blocks:
        m = np.zeros((1, model.n_readout), dtype=bool)
        m[0, block] = True
        resp = model.forward_batch(x0[None, :], perturbed_col=None, mask=m)[0]
        base_acc[block] += resp[block]
        coverage[block] += 1
    base = base_acc / np.maximum(coverage, 1)

    delta = np.zeros((cols.size, len(grid), model.n_readout), dtype=np.float64)
    y_pert = np.zeros_like(delta)
    for pi, col in enumerate(cols):
        for vi, v in enumerate(values_native):
            acc = np.zeros(model.n_readout, dtype=np.float64)
            for block in blocks:
                m = np.zeros((1, model.n_readout), dtype=bool)
                m[0, block] = True
                x = x0.copy()
                x[int(col)] = float(v)
                resp = model.forward_batch(x[None, :], perturbed_col=int(col), mask=m)[0]
                acc[block] += resp[block]
            resp_mean = acc / np.maximum(coverage, 1)
            y_pert[pi, vi] = resp_mean
            delta[pi, vi] = resp_mean - base
        if (pi + 1) % 10 == 0:
            print(f"    masked gene {pi + 1}/{cols.size}", flush=True)

    y_base = np.repeat(base[None, None, :], cols.size, axis=0).repeat(len(grid), axis=1)
    forms = normalize_forms(delta, y_pert, y_base, model)
    mdir = paths.ensure(out_dir / "anchor=mean_control_masked")
    np.savez(
        mdir / "vvp.npz",
        **forms,
        perturbed_gene_id=subset.gene_id.to_numpy().astype(str),
        readout_gene_id=readout_ids,
        value_grid_cp10k=np.asarray(grid, dtype=np.float64),
    )
    zero_i = list(grid).index(0.0)
    zero_forms = normalize_forms(delta[:, zero_i], y_pert[:, zero_i], y_base[:, zero_i], model)
    np.savez(mdir / "zero_out.npz", **zero_forms,
             perturbed_gene_id=subset.gene_id.to_numpy().astype(str))
    return {
        "ran": True,
        "mask_spec": spec,
        "mask_rate": rate,
        "n_blocks_per_cycle": int(np.ceil(1.0 / rate)),
        "cycles": C.MASK_MIN_COVERAGE,
        "coverage_per_gene": {
            "min": int(coverage.min()),
            "median": float(np.median(coverage)),
            "max": int(coverage.max()),
        },
        "n_perturbed": int(cols.size),
        "value_grid_cp10k": list(grid),
        "seconds": round(time.time() - started, 1),
        "note": "responses are read only at masked positions, averaged over cycles; the "
        "unmasked block at the same anchor is the comparison",
    }


def compare_context_widths(model, data: dict, readout: pd.DataFrame, perturbed: pd.DataFrame, out_dir: Path) -> dict:
    """scGPT only: how far do responses move between three supplied-set widths?"""
    subset = perturbed.iloc[: C.N_CONTEXT_COMPARE_GENES]
    cols = subset["readout_col"].to_numpy(dtype=np.int64)
    control = readout["pooled_control_cp10k"].to_numpy(dtype=np.float64)
    universe = data["universe"]
    uni_control = (
        universe.merge(
            readout[["gene_id", "pooled_control_cp10k"]], on="gene_id", how="left"
        )["pooled_control_cp10k"]
        .fillna(C.RUN1_FLOOR_CP10K / 2)
        .to_numpy(dtype=np.float64)
    )
    out: dict[str, np.ndarray] = {}
    for mode in ("readout", "universe", "tiled"):
        model.configure(
            readout, control, model.cell_line, supplied_mode=mode, universe_anchor_cp10k=uni_control
        )
        x0 = np.asarray(model.to_native(control), dtype=np.float32)
        deltas = []
        for col in cols:
            x = x0.copy()
            x[int(col)] = 0.0
            resp = model.forward_batch(np.stack([x0, x]), perturbed_col=int(col))
            deltas.append(resp[1] - resp[0])
        out[mode] = np.stack(deltas)
        print(f"    context {mode}: |delta| median {np.median(np.abs(out[mode])):.4g}", flush=True)
    pairs = {}
    for a, b in (("readout", "universe"), ("readout", "tiled"), ("universe", "tiled")):
        per = [
            float(np.corrcoef(out[a][i], out[b][i])[0, 1])
            for i in range(cols.size)
            if np.std(out[a][i]) > 0 and np.std(out[b][i]) > 0
        ]
        rel = [
            float(2 * np.linalg.norm(out[a][i] - out[b][i]) /
                  (np.linalg.norm(out[a][i]) + np.linalg.norm(out[b][i])))
            for i in range(cols.size)
        ]
        pairs[f"{a}_vs_{b}"] = {
            "median_pearson": float(np.median(per)),
            "median_relative_l2": float(np.median(rel)),
        }
    np.savez(out_dir / "context_comparison.npz", **out,
             perturbed_gene_id=subset.gene_id.to_numpy().astype(str))
    # restore the production layout
    model.configure(readout, control, model.cell_line, supplied_mode=C.SCGPT_PRODUCTION_MODE)
    return {
        "n_genes": int(cols.size),
        "form": "zero-out response difference at the mean control anchor",
        "widths": {
            "readout": int(readout.shape[0]),
            "universe": int(universe.shape[0]),
            "tiled": model.TILE_LIMIT,
        },
        "agreement": pairs,
        "production_mode": C.SCGPT_PRODUCTION_MODE,
    }


def untiled_exact_check(model, readout: pd.DataFrame, perturbed: pd.DataFrame, step: float) -> dict:
    """scGPT only: the exact derivative the production layout cannot provide.

    A tiled query is several forward passes, so it is not one differentiable
    function of the input and the production gradient is a central difference.
    The same model in the untiled layout does admit forward-mode differentiation,
    so the two are compared there on a small subset: it is evidence about the
    difference estimator, measured on the same weights.
    """
    control = readout["pooled_control_cp10k"].to_numpy(dtype=np.float64)
    cols = perturbed["readout_col"].to_numpy(dtype=np.int64)[: C.N_UNTILED_CHECK_GENES]
    model.configure(readout, control, model.cell_line, supplied_mode="readout")
    x0 = np.asarray(model.to_native(control), dtype=np.float32)
    values = np.asarray(model.to_native(np.asarray(C.VALUE_GRID_CP10K, dtype=np.float64)))
    rows = []
    for vi, v in enumerate(values):
        exact, fd = [], []
        for col in cols:
            x = x0.copy()
            x[int(col)] = float(v)
            jvp = model.exact_jvp(x, int(col))
            if jvp is None:
                model.configure(readout, control, model.cell_line,
                                supplied_mode=C.SCGPT_PRODUCTION_MODE)
                return {"ran": False, "reason": getattr(model, "_jvp_error", "unavailable")}
            lo = max(float(v) - step, NATIVE_FLOOR)
            xa, xb = x0.copy(), x0.copy()
            xa[int(col)], xb[int(col)] = lo, float(v) + step
            resp = model.forward_batch(np.stack([xa, xb]), perturbed_col=int(col))
            exact.append(jvp)
            fd.append((resp[1] - resp[0]) / max(float(v) + step - lo, 1e-12))
        a, b = np.concatenate(exact), np.concatenate(fd)
        denom = np.linalg.norm(a) + np.linalg.norm(b)
        rows.append(
            {
                "value_cp10k": float(C.VALUE_GRID_CP10K[vi]),
                "one_sided": bool(float(v) - step < NATIVE_FLOOR),
                "pearson": float(np.corrcoef(a, b)[0, 1]),
                "relative_l2": float(2 * np.linalg.norm(a - b) / denom) if denom else None,
            }
        )
    model.configure(readout, control, model.cell_line, supplied_mode=C.SCGPT_PRODUCTION_MODE)
    two_sided = [r for r in rows if not r["one_sided"]]
    return {
        "ran": True,
        "layout": "readout (untiled), the layout that admits forward-mode autodiff",
        "n_genes": int(cols.size),
        "step": step,
        "per_value": rows,
        "median_pearson_two_sided": float(np.median([r["pearson"] for r in two_sided])),
        "median_relative_l2_two_sided": float(
            np.median([r["relative_l2"] for r in two_sided])
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(paths.MODELS))
    ap.add_argument("--cell-line", required=True, choices=list(paths.CELL_LINES))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--n-perturbed", type=int, default=None)
    ap.add_argument("--n-cells-anchor", type=int, default=None)
    ap.add_argument("--n-masked", type=int, default=None)
    ap.add_argument("--n-step-sweep", type=int, default=None)
    ap.add_argument("--exact-derivative-genes", type=int, default=None)
    ap.add_argument("--anchors", nargs="*", default=list(C.ANCHORS))
    ap.add_argument(
        "--full-scope",
        action="store_true",
        help="every in-scope perturbation at every anchor and in the masked variant; "
        "writes to the features_full_scope tree so run 2's subset blocks stay intact",
    )
    ap.add_argument("--supplied-mode", default=None)
    ap.add_argument("--no-masked", action="store_true")
    ap.add_argument("--no-context-compare", action="store_true")
    ap.add_argument("--image", default="unrecorded")
    args = ap.parse_args(argv)

    if args.n_step_sweep:
        C.N_STEP_SWEEP_GENES = int(args.n_step_sweep)
    if args.exact_derivative_genes is not None:
        C.N_EXACT_DERIVATIVE_GENES = int(args.exact_derivative_genes)
    # 0 means "every gene in scope", so these read `is not None` rather than being
    # treated as falsy
    base = C.FULL_SCOPE if args.full_scope else C.SCOPE
    scope = C.Scope(
        n_perturbed=base.n_perturbed if args.n_perturbed is None else args.n_perturbed,
        n_perturbed_cells_anchor=(
            base.n_perturbed_cells_anchor if args.n_cells_anchor is None else args.n_cells_anchor
        ),
        n_perturbed_masked=base.n_perturbed_masked if args.n_masked is None else args.n_masked,
    )
    cache = Path(args.cache) if args.cache else paths.checkpoint_cache()
    default_out = (
        paths.features_full_dir(args.model, args.cell_line)
        if args.full_scope
        else paths.features_dir(args.model, args.cell_line)
    )
    out = Path(args.out) if args.out else default_out
    try:
        run(
            args.model,
            args.cell_line,
            cache,
            out,
            scope,
            device=args.device,
            anchors=tuple(args.anchors),
            batch=args.batch,
            supplied_mode=args.supplied_mode,
            image=args.image,
            do_masked=not args.no_masked,
            do_context_compare=not args.no_context_compare,
        )
    except Exception:
        paths.ensure(out)
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
