"""What full coverage would cost for one model, measured rather than guessed.

Three questions about one model, answered in one short job, so a coverage
decision is made against a number:

**throughput** --- seconds per forward pass at several batch sizes at the
production anchor. The extraction driver issues one call per perturbed gene
carrying that gene's whole value grid, so "batching across perturbed genes" means
raising that call's batch. Whether it helps is a property of the model's sequence
length and the GPU, not something to assume.

**projection** --- the forward passes full coverage needs, from the same counting
the driver actually does, priced at the measured rate. The projection is checked
against the block already extracted: predicting a run that has happened is the
only way to know the arithmetic is right.

**forward mode** --- whether ``torch.func.jvp`` runs through this build at all,
and if it does, how far the exact derivative sits from the central difference the
block used. Run 1's AIDO.Cell contract recorded forward-mode autodiff as
unavailable in a substitution note while the same contract's
``jacobian_available`` field asserted the opposite; only an attempt settles it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .. import config as C
from .. import paths
from ..models import queries as Q
from ..models.base import environment_manifest
from .extract import NATIVE_FLOOR, load_inputs
from .scope import in_scope_perturbed, select_perturbed


def time_forward(
    model, x0: np.ndarray, col: int, batch_size: int, n_batches: int = 3,
    autocast_dtype: str | None = None,
) -> dict:
    """Seconds per forward pass at one batch size, after a warm-up batch.

    A lower-precision measurement runs under autocast rather than casting the
    weights, which is both the realistic way to speed this up and the only one
    that does not break an adapter that hands the model float32 inputs.
    """
    rows = np.repeat(x0[None, :], batch_size, axis=0).astype(np.float32)
    rows[:, col] = np.linspace(0.0, 4.0, batch_size)
    model.batch_size = int(batch_size)
    ctx = (
        torch.autocast("cuda", dtype=getattr(torch, autocast_dtype))
        if autocast_dtype
        else contextlib.nullcontext()
    )
    with ctx:
        model.forward_batch(rows, perturbed_col=int(col))  # warm up
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.time()
        for _ in range(n_batches):
            model.forward_batch(rows, perturbed_col=int(col))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    seconds = (time.time() - started) / n_batches
    return {
        "batch_size": int(batch_size),
        "autocast": autocast_dtype or "none (float32)",
        "seconds_per_call": round(seconds, 4),
        "seconds_per_forward": round(seconds / batch_size, 5),
        "n_calls_timed": n_batches,
        "peak_gpu_gib": (
            round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
        ),
    }


def forward_passes_needed(
    n_perturbed: int, n_cells_anchor: int, n_masked: int, n_cells: int, n_values: int,
    n_masked_values: int, mask_blocks: int, n_step_genes: int, n_steps: int, n_anchors_single: int,
) -> dict:
    """The driver's own arithmetic, so a projection prices the real work.

    One perturbed gene at one anchor cell costs ``1 + 3 * n_values`` passes: the
    anchor row once, then the value and its two finite-difference neighbours at
    each grid value.
    """
    per_gene_cell = 1 + 3 * n_values
    single = n_anchors_single * n_perturbed * per_gene_cell
    cells = n_cells_anchor * n_cells * per_gene_cell
    masked = mask_blocks * (1 + n_masked * n_masked_values)
    sweep = (n_anchors_single + 1) * n_steps * n_step_genes * 2
    return {
        "per_gene_per_cell": per_gene_cell,
        "single_cell_anchors": int(single),
        "individual_cells_anchor": int(cells),
        "masked_variant": int(masked),
        "step_sweeps": int(sweep),
        "total": int(single + cells + masked + sweep),
    }


def probe_forward_mode(model, x0: np.ndarray, cols: np.ndarray, step: float,
                       grid_cp10k: tuple[float, ...]) -> dict:
    """Does forward-mode autodiff run, and does it agree with the difference?"""
    values = np.asarray(model.to_native(np.asarray(grid_cp10k, dtype=np.float64)))
    first, reason = None, None
    per_value = []
    for vi, v in enumerate(values):
        exact, fd = [], []
        for col in cols:
            x = x0.copy()
            x[int(col)] = float(v)
            jvp, why = model.forward_mode_probe(x, int(col))
            if jvp is None:
                return {
                    "available": False,
                    "reason": why,
                    "n_genes_attempted": 1,
                    "note": "the forward-mode route does not run through this build, so the "
                    "central difference at the selected step is the only derivative available "
                    "and the block's gradient features have no exact cross-check",
                }
            lo = max(float(v) - step, NATIVE_FLOOR)
            xa, xb = x0.copy(), x0.copy()
            xa[int(col)], xb[int(col)] = lo, float(v) + step
            resp = model.forward_batch(np.stack([xa, xb]).astype(np.float32), perturbed_col=int(col))
            exact.append(np.asarray(jvp, dtype=np.float64))
            fd.append((resp[1] - resp[0]).astype(np.float64) / max(float(v) + step - lo, 1e-12))
        a, b = np.concatenate(exact), np.concatenate(fd)
        denom = np.linalg.norm(a) + np.linalg.norm(b)
        per_value.append(
            {
                "value_cp10k": float(grid_cp10k[vi]),
                "one_sided": bool(float(v) - step < NATIVE_FLOOR),
                "pearson": float(np.corrcoef(a, b)[0, 1])
                if a.std() > 0 and b.std() > 0 else None,
                "relative_l2": float(2 * np.linalg.norm(a - b) / denom) if denom else None,
                "exact_l2": float(np.linalg.norm(a)),
                "finite_difference_l2": float(np.linalg.norm(b)),
            }
        )
        first = first or "ran"
    two_sided = [r for r in per_value if not r["one_sided"] and r["pearson"] is not None]
    return {
        "available": True,
        "reason": reason,
        "n_genes": int(cols.size),
        "step": float(step),
        "per_value": per_value,
        "median_pearson_two_sided": float(np.median([r["pearson"] for r in two_sided]))
        if two_sided else None,
        "median_relative_l2_two_sided": float(np.median([r["relative_l2"] for r in two_sided]))
        if two_sided else None,
        "note": "the exact forward-mode derivative against the central difference at the step "
        "the block selected, on a small gene sample at the production anchor; a value whose "
        "difference would cross the input floor is one-sided and excluded from the medians",
    }


def run(
    model_name: str,
    cell_lines: tuple[str, ...],
    cache: Path,
    out_dir: Path,
    batch_sizes: tuple[int, ...],
    device: str = "cuda",
    n_exact_genes: int = 8,
    observed: dict | None = None,
    dtypes: tuple[str, ...] = ("float32",),
    image: str = "unrecorded",
) -> dict:
    out = {"model": model_name, "batch_sizes": list(batch_sizes), "per_cell_line": {}}
    first = True
    for cell_line in cell_lines:
        data = load_inputs(model_name, cell_line)
        readout = data["readout"]
        subset = select_perturbed(data["perturbations"], readout, C.SCOPE)
        available = in_scope_perturbed(data["perturbations"], readout, C.SCOPE)
        model = Q.build(model_name, cache, data["universe"], data["vocab"], device=device)
        model.load()
        control = readout["pooled_control_cp10k"].to_numpy(dtype=np.float64)
        kwargs = {"supplied_mode": C.SCGPT_PRODUCTION_MODE} if model_name == "scgpt" else {}
        layout = model.configure(readout, control, cell_line, **kwargs)
        x0 = np.asarray(model.to_native(control), dtype=np.float32)
        col = int(subset.readout_col.iloc[0])
        default_batch = int(type(model).batch_size)

        throughput = []
        for bs in batch_sizes:
            # a batch that does not fit is itself an answer to "can batching bring the
            # cost down", so it is recorded rather than allowed to end the job
            try:
                row = time_forward(model, x0, col, bs)
            except Exception as exc:  # noqa: BLE001
                row = {"batch_size": int(bs), "error": f"{type(exc).__name__}: {exc}"[:300]}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            throughput.append(row)
            print(f"[{model_name}/{cell_line}] batch {bs}: {row}", flush=True)
        model.batch_size = default_batch
        throughput_ok = [r for r in throughput if "seconds_per_forward" in r]
        if not throughput_ok:
            raise RuntimeError(f"no batch size completed a forward pass: {throughput}")

        alt_dtype = {}
        for dt in dtypes:
            if dt == "float32":
                continue
            try:
                rows = []
                for bs in batch_sizes:
                    try:
                        rows.append(time_forward(model, x0, col, bs, autocast_dtype=dt))
                    except Exception as exc:  # noqa: BLE001
                        rows.append({"batch_size": int(bs), "error": f"{type(exc).__name__}: {exc}"[:300]})
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                ok = [r for r in rows if "seconds_per_forward" in r]
                alt_dtype[dt] = {
                    "throughput": rows,
                    "best_seconds_per_forward": min(r["seconds_per_forward"] for r in ok) if ok else None,
                    "warning": "not a like-for-like extension: the block's numerics change, so a "
                    "partly-float32 and partly-lower-precision feature set would not be one "
                    "block. Recorded as the cost of a deliberate re-extraction, not of a top-up.",
                }
                print(f"[{model_name}/{cell_line}] autocast {dt}: {rows}", flush=True)
            except Exception as exc:  # noqa: BLE001
                alt_dtype[dt] = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                model.batch_size = default_batch

        best = min(throughput_ok, key=lambda r: r["seconds_per_forward"])
        n_values = len(C.VALUE_GRID_CP10K)
        spec = model.mask_spec()
        mask_blocks = (
            int(np.ceil(1.0 / float(spec["rate"]))) * C.MASK_MIN_COVERAGE if spec else 0
        )
        n_full = int(available.shape[0])
        counts = {
            "subset": forward_passes_needed(
                n_perturbed=int(subset.shape[0]),
                n_cells_anchor=int(subset.in_cells_anchor.sum()),
                n_masked=int(subset.in_masked.sum()),
                n_cells=C.N_INDIVIDUAL_CELLS, n_values=n_values,
                n_masked_values=len(C.MASKED_VALUE_GRID_CP10K), mask_blocks=mask_blocks,
                n_step_genes=C.N_STEP_SWEEP_GENES, n_steps=len(C.STEP_GRID[model_name]),
                n_anchors_single=2,
            ),
            "full": forward_passes_needed(
                n_perturbed=n_full, n_cells_anchor=n_full, n_masked=n_full,
                n_cells=C.N_INDIVIDUAL_CELLS, n_values=n_values,
                n_masked_values=len(C.MASKED_VALUE_GRID_CP10K), mask_blocks=mask_blocks,
                n_step_genes=C.N_STEP_SWEEP_GENES, n_steps=len(C.STEP_GRID[model_name]),
                n_anchors_single=2,
            ),
        }
        at_default = next(
            (r for r in throughput_ok if r["batch_size"] == default_batch), throughput_ok[0]
        )
        obs = (observed or {}).get(cell_line)
        # the three anchors only, which is what the observed seconds in the shipped
        # manifest cover, so predicted and observed price the same work
        anchors_only_subset = (
            counts["subset"]["single_cell_anchors"] + counts["subset"]["individual_cells_anchor"]
        )
        anchors_only_full = (
            counts["full"]["single_cell_anchors"] + counts["full"]["individual_cells_anchor"]
        )
        projection = {
            "n_in_scope": n_full,
            "n_subset": int(subset.shape[0]),
            "forward_passes": counts,
            "rate_used_seconds_per_forward": at_default["seconds_per_forward"],
            "rate_batch_size": at_default["batch_size"],
            "best_rate_seconds_per_forward": best["seconds_per_forward"],
            "best_batch_size": best["batch_size"],
            "batching_speedup": round(
                at_default["seconds_per_forward"] / best["seconds_per_forward"], 3
            ),
            "predicted_subset_three_anchor_seconds": round(
                anchors_only_subset * at_default["seconds_per_forward"]
            ),
            "observed_subset_three_anchor_seconds": obs,
            "prediction_error_vs_observed": (
                round(anchors_only_subset * at_default["seconds_per_forward"] / obs - 1.0, 3)
                if obs else None
            ),
            "projected_full_three_anchor_seconds": round(
                anchors_only_full * at_default["seconds_per_forward"]
            ),
            "projected_full_three_anchor_seconds_best_batch": round(
                anchors_only_full * best["seconds_per_forward"]
            ),
            "projected_full_everything_seconds": round(
                counts["full"]["total"] * at_default["seconds_per_forward"]
            ),
            "projected_full_everything_seconds_best_batch": round(
                counts["full"]["total"] * best["seconds_per_forward"]
            ),
        }
        print(f"[{model_name}/{cell_line}] projection {json.dumps(projection)}", flush=True)

        entry = {
            "n_readout": int(readout.shape[0]),
            "layout": layout.as_dict(),
            "default_batch_size": default_batch,
            "throughput": throughput,
            "alternate_dtypes": alt_dtype,
            "projection": projection,
        }
        if first:
            step = float(C.STEP_GRID[model_name][0])
            cols = subset.readout_col.to_numpy(dtype=np.int64)[:n_exact_genes]
            entry["forward_mode_probe"] = probe_forward_mode(
                model, x0, cols, step, C.VALUE_GRID_CP10K
            )
            print(
                f"[{model_name}/{cell_line}] forward mode "
                f"{json.dumps(entry['forward_mode_probe'])[:600]}",
                flush=True,
            )
            first = False
        out["per_cell_line"][cell_line] = entry
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out["environment"] = environment_manifest(model_name, image, {"device": device})
    out_dir = paths.ensure(out_dir)
    (out_dir / f"cost_probe_{model_name}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {out_dir / f'cost_probe_{model_name}.json'}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(paths.MODELS))
    ap.add_argument("--cell-lines", nargs="*", default=list(paths.CELL_LINES))
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--dtypes", nargs="*", default=["float32", "bfloat16"])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-exact-genes", type=int, default=8)
    ap.add_argument(
        "--observed",
        default=None,
        help="JSON of {cell_line: observed three-anchor seconds} from the shipped manifest, so "
             "the projection is checked against a run that already happened",
    )
    ap.add_argument("--image", default="unrecorded")
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else paths.report_dir() / "cost_probe"
    try:
        run(
            args.model,
            tuple(args.cell_lines),
            Path(args.cache) if args.cache else paths.checkpoint_cache(),
            out,
            tuple(args.batch_sizes),
            device=args.device,
            n_exact_genes=args.n_exact_genes,
            observed=json.loads(args.observed) if args.observed else None,
            dtypes=tuple(args.dtypes),
            image=args.image,
        )
    except Exception:
        paths.ensure(out)
        (out / f"failure_{args.model}.txt").write_text(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
