"""Drive run 2's extractor against the fine-tuned model's query adapter.

Throwaway: it checks the wiring between run 2's ``extract_anchor`` and the
adapter -- shapes, forms, the forward-mode derivative, the perturbation-flag
query -- on a handful of genes, before a real job spends an hour to find out.
No bundle and no cells needed: the perturbed set is a slice of the read-out
table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scfm_ft import config as C  # noqa: E402
from scfm_ft import paths  # noqa: E402
from scfm_ft.run_arm import extract_stage, substituted_table  # noqa: E402
from scfm_ft.scgpt_pert import checkpoint, model as M  # noqa: E402


def main() -> int:
    line = sys.argv[1] if len(sys.argv) > 1 else "hepg2"
    n_pert = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    readout = pd.read_parquet(paths.readout_table(line))
    cache = paths.checkpoint_cache()
    checkpoint.fetch(cache)

    take = np.linspace(0, readout.shape[0] - 1, n_pert).astype(np.int64)
    targets = {
        "self_col": take,
        "pert_gene_id": readout["gene_id"].to_numpy().astype(str)[take],
        "pert_target": readout["gene_name"].to_numpy().astype(str)[take],
        "readout_gene_id": readout["gene_id"].to_numpy().astype(str),
        "readout_gene_name": readout["gene_name"].to_numpy().astype(str),
    }
    pos = readout["scgpt_position"].to_numpy(dtype=np.int64)

    for arm_name in ("own_trainable", "rand_frozen"):
        arm = C.ARMS_BY_NAME[arm_name]
        table = None
        if arm.table != "scgpt":
            fake = np.random.default_rng(0).standard_normal((readout.shape[0], 2560)).astype(np.float32)
            np.savez("/tmp/fake_esm2.npz", embedding=fake,
                     gene_id=targets["readout_gene_id"], mapped=np.ones(readout.shape[0], bool))
            table, rep = substituted_table(arm, Path("/tmp/fake_esm2.npz"), targets["readout_gene_id"])
            print(f"table report: {json.dumps(rep)}", flush=True)
        model, report = M.build_arm_model(arm, cache, table, pos, device="cuda")
        print(f"\n=== {arm_name}: {report['trainable_parameters']} trainable of "
              f"{report['total_parameters']} ===", flush=True)
        t0 = time.time()
        b = extract_stage(model, "final", arm_name, readout, targets, "cuda")
        print(f"extraction of {n_pert} genes in {time.time() - t0:.0f}s", flush=True)
        for form, arr in b["value_forms"].items():
            a = np.asarray(arr)
            print(f"  {form}: shape {a.shape} finite {np.isfinite(a).all()} "
                  f"mean {a.mean():.4g} sd {a.std():.4g}", flush=True)
        fa = np.asarray(b["flag_forms"]["zero_out"])
        print(f"  pert_flag zero_out: shape {fa.shape} finite {np.isfinite(fa).all()} "
              f"sd {fa.std():.4g}", flush=True)
        print(f"  gene table: {np.asarray(b['gene_table']).shape} — {b['gene_table_note']}", flush=True)
        d = b["diagnostics"]
        print("  diagnostics:", json.dumps({
            "gdt_source": d.get("gdt_source"),
            "forward_mode": d.get("forward_mode"),
            "pert_flag_delta_abs_median": d.get("pert_flag_delta_abs_median"),
            "pert_flag_self_delta_median": d.get("pert_flag_self_delta_median"),
            "seconds": d.get("seconds"),
        }, indent=2), flush=True)
        ev = d.get("exact_vs_finite_difference")
        if ev:
            print("  exact vs finite difference:", json.dumps({
                "median_pearson_two_sided": ev.get("median_pearson_two_sided"),
                "median_relative_l2_two_sided": ev.get("median_relative_l2_two_sided"),
            }), flush=True)
        del model
        import torch

        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
